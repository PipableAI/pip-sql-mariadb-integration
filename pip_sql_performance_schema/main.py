import json
from pprint import pprint
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import pickle
import torch
import torch.nn.functional as F
import pandas as pd
import pymysql
import huggingface_hub as hf_hub
import os

class PipSQLPerformanceSchema:

    def __init__(self, 
                 embed_model_id="Salesforce/codet5p-110m-embedding", 
                 sql_model_id="PipableAI/pipSQL-mariadb-performance-schema-1.3b", 
                 device="cuda", 
                 hf_token="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",
                 sql_host="20.163.176.230",
                 sql_user="stan",
                 sql_password="123456",
                 ):
        self.embed_model_id = embed_model_id
        self.sql_model_id = sql_model_id
        self.device = device
        self.hf_token = hf_token
        self.sql_host = sql_host
        self.sql_user = sql_user
        self.sql_password = sql_password

        hf_hub.login(hf_token)
        print("Loading Models...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_id, trust_remote_code=True)
        self.sql_tokenizer = AutoTokenizer.from_pretrained(sql_model_id)
        self.sql_model = AutoModelForCausalLM.from_pretrained(sql_model_id, torch_dtype=torch.bfloat16).to(device)
        self.embed_model = AutoModel.from_pretrained(embed_model_id, trust_remote_code=True)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompt_path = os.path.join(self.script_dir, "data")
        self.maria_db_performance_schema_json_path = os.path.join(self.prompt_path, "mariadb_performance_schema.json")
        self.table_embeddings_path = os.path.join(self.prompt_path, "table_embeddings.pkl")
        self.maria_db_ddls_path = os.path.join(self.prompt_path, "maria_db_ddls.json")


        with open(self.maria_db_performance_schema_json_path, "r") as f:
            self.performance_schema = json.loads(f.read())

        with open(self.table_embeddings_path, "rb") as f:
            self.table_embeddings = pickle.load(f)

        with open(self.maria_db_ddls_path, "r") as f:
            self.maria_db_performance_schema_ddls = json.loads(f.read())
        
    def execute_query(self, query):
        connection = pymysql.connect(
            host=self.sql_host,
            user=self.sql_user,
            password=self.sql_password,
            database="performance_schema",
            port=3306,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(result, columns=columns)
                return df
    
        except Exception as e:
            return str(e)
        finally:
            connection.close()

    def generate_embedding(self, input):
        encoded_input = self.embed_tokenizer(input, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)[0]
    
        return model_output

    def get_table_score(self, input, table):
        input_embedding = self.generate_embedding(input).unsqueeze(0)
        ddl_embedding = self.table_embeddings[table]['ddl_embedding'].unsqueeze(0)

        table_similarity = F.cosine_similarity(input_embedding, ddl_embedding, dim=1)
        column_similarities = [F.cosine_similarity(input_embedding, col_embedding.unsqueeze(0), dim=1) for col, col_embedding in self.table_embeddings[table]['column_embeddings'].items()]

        #Column relevance score
        high_relevance_weight=0.3
        table_context_weight=0.7

        max_column_score = max(column_similarities) if column_similarities else 0
        table_score = max_column_score * high_relevance_weight + table_similarity * table_context_weight
        return table_score

    def choose_tables(self, table_scores):
        table_scores = [(table, score.item()) for table, score in table_scores]
        table_scores[0] = (table_scores[0][0], table_scores[0][1], 0)

        for i in range(1, len(table_scores)):
            table_scores[i] = (table_scores[i][0], table_scores[i][1], (table_scores[i-1][1] - table_scores[i][1])*100/table_scores[i-1][1])

        chosen_tables = []
        for table, score, drop in table_scores:
            if drop < 7.5:
                chosen_tables.append(table)
            else:
                break
    
        return chosen_tables[:10]
    
    def ask_pip(self, question):
        table_scores = table_scores = {table: self.get_table_score(question, table) for table in self.performance_schema.keys()}
        table_scores = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        chosen_tables = self.choose_tables(table_scores)
        schema = '\n'.join([self.maria_db_performance_schema_ddls[table] for table in chosen_tables])

        prompt=f"""Generate the SQL query for SkySQL performance schema for the following question.
<example>
--question: What are the top 10 most frequently used queries/statements?
--sql: SELECT DIGEST_TEXT, COUNT(*) as frequency FROM performance_schema.events_statements_summary_by_digest GROUP BY DIGEST_TEXT ORDER BY frequency DESC LIMIT 10;
</example>
<schema>
{schema}
</schema>
<question>
{question}
</question>
<sql>
    """
        inputs = self.sql_tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.sql_model.generate(**inputs, max_new_tokens=500)
        output = self.sql_tokenizer.decode(outputs[0])
        del inputs
        del outputs
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return output.split('<sql>')[1].split('</sql>')[0].strip()
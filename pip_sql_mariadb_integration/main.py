import json
from pprint import pprint
import pickle
import torch
import torch.nn.functional as F
import pandas as pd
import pymysql
import os
import requests

class PipSQLMariaDBIntegration:

    def __init__(self, 
                 embed_model_id="BAAI/bge-m3", 
                 sql_model_id="PipableAI/pipSQL-mariadb-performance-schema-1.3b", 
                 device="cuda", 
                 hf_token="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",
                 sql_host="20.163.176.230",
                 sql_user="stan",
                 sql_password="123456",
                 model_inference_url="https://playground.pipable.ai/infer"
                 ):
        self.embed_model_id = embed_model_id
        self.sql_model_id = sql_model_id
        self.device = device
        self.hf_token = hf_token
        self.sql_host = sql_host
        self.sql_user = sql_user
        self.sql_password = sql_password
        self.model_inference_url = model_inference_url

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompt_path = os.path.join(self.script_dir, "data")
        self.performance_schema_json_path = os.path.join(self.prompt_path, "performance_schema.json")
        self.performance_schema_embeddings_path = os.path.join(self.prompt_path, "performance_schema_embeddings.pkl")
        self.performance_schema_ddls_path = os.path.join(self.prompt_path, "performance_schema_ddls.json")
        self.information_schema_json_path = os.path.join(self.prompt_path, "information_schema.json")
        self.information_schema_embeddings_path = os.path.join(self.prompt_path, "information_schema_embeddings.pkl")
        self.information_schema_ddls_path = os.path.join(self.prompt_path, "information_schema_ddls.json")


        with open(self.performance_schema_json_path, "r") as f:
            self.performance_schema = json.loads(f.read())

        with open(self.performance_schema_embeddings_path, "rb") as f:
            self.performance_schema_embeddings = pickle.load(f)

        with open(self.performance_schema_ddls_path, "r") as f:
            self.performance_schema_ddls = json.loads(f.read())
        
        with open(self.information_schema_json_path, "r") as f:
            self.information_schema = json.loads(f.read())
        
        with open(self.information_schema_embeddings_path, "rb") as f:
            self.information_schema_embeddings = pickle.load(f)
        
        with open(self.information_schema_ddls_path, "r") as f:
            self.information_schema_ddls = json.loads(f.read())
        
    def execute_query(self, query, database):
        connection = pymysql.connect(
            host=self.sql_host,
            user=self.sql_user,
            password=self.sql_password,
            database=database,
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
    
    def query_sql_model(self, prompt, max_new_tokens=250):
        payload = {
                "model_name": self.sql_model_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            }
        response = requests.request(
            method="POST", url=self.model_inference_url, data=payload, timeout=120
        )
        if response.status_code == 200:
            return json.loads(response.text)["response"]
        else:
            raise Exception(f"Error generating response using {self.model_inference_url}.")
    
    def query_embed_model(self, prompt):
        payload = {
                "model_name": self.embed_model_id,
                "prompt": prompt,
                "model_type": "embed"
            }
        response = requests.request(
            method="POST", url=self.model_inference_url, data=payload, timeout=120
        )
        if response.status_code == 200:
            res = json.loads(json.loads(response.text)['response'])
            return torch.tensor(res)
        else:
            raise Exception(f"Error generating response using {self.model_inference_url}.")


    def get_table_score(self, input_embedding, table, embeddings):
        input_embedding = input_embedding.unsqueeze(0)
        ddl_embedding = embeddings[table]['ddl_embedding'].unsqueeze(0)

        table_similarity = F.cosine_similarity(input_embedding, ddl_embedding, dim=1)
        column_similarities = [F.cosine_similarity(input_embedding, col_embedding.unsqueeze(0), dim=1) for col, col_embedding in embeddings[table]['column_embeddings'].items()]

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
            if drop <= 5.0:
                chosen_tables.append(table)
            else:
                break
    
        return chosen_tables[:10]
    
    def ask_pip_performance_schema(self, question, print_table=False):
        input_embedding = self.query_embed_model(question)
        table_scores = {table: self.get_table_score(input_embedding, table, self.performance_schema_embeddings) for table in self.performance_schema.keys()}
        table_scores = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        chosen_tables = self.choose_tables(table_scores)
        if print_table:
            print(chosen_tables)
        schema = '\n'.join([self.performance_schema_ddls[table] for table in chosen_tables])

        prompt=f"""Generate the SQL query for SkySQL performance schema for the following question.
<instructions>
1. Use only the tables that are present in the schema
2. Use only the columns that are present in the schema
3. If using all columns, use *
4. In performance schema queries means DIGEST_TEXT
<example>
--question: What are the top 10 most frequently used queries/statements?
--sql: SELECT DIGEST_TEXT, COUNT(*) as frequency FROM events_statements_summary_by_digest GROUP BY DIGEST_TEXT ORDER BY frequency DESC LIMIT 10;

--question: List all details about the users?
--sql: SELECT * FROM users;

--question: List the top 10 queries with the with the most CPU wait time;
--sql: SELECT DIGEST_TEXT, SUM(SUM_TIMER_WAIT) as cpu_time FROM events_statements_summary_by_digest GROUP BY DIGEST_TEXT ORDER BY cpu_time DESC LIMIT 10;

--question: 
</example>
<schema>
{schema}
</schema>
<question>
{question}
</question>
<sql>
    """
        max_new_tokens = 500
        output = self.query_sql_model(prompt, max_new_tokens)
        return output.split('<sql>')[1].split('</sql>')[0].strip()
    
    def ask_pip_information_schema(self, question, print_table=False):
        input_embedding = self.query_embed_model(question)
        table_scores = {table: self.get_table_score(input_embedding, table, self.information_schema_embeddings) for table in self.information_schema.keys()}
        table_scores = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        chosen_tables = self.choose_tables(table_scores)
        if print_table:
            print(chosen_tables)
        schema = '\n'.join([self.information_schema_ddls[table] for table in chosen_tables])

        prompt=f"""Generate the SQL query for SkySQL information schema for the following question. 
<instructions>
1. In Privileges tables, user is GRANTEE
2. Use only the tables that are present in the schema
3. Use only the columns that are present in the schema
4. If using all columns, use *
</instructions>
<example>
--question: List the character set and collation for each database;
--sql: SELECT SCHEMA_NAME, DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME FROM information_schema.SCHEMATA;

--question: List the index used by the tables of the database db1;
--sql: SELECT TABLE_NAME, INDEX_NAME FROM STATISTICS WHERE TABLE_SCHEMA='db1';
</example>
<schema>
{schema}
</schema>
<question>
{question}
</question>
<sql>
"""     
        max_new_tokens = 500
        output = self.query_sql_model(prompt, max_new_tokens)
        return output.split('<sql>')[1].split('</sql>')[0].strip()
from pip_sql_performance_schema import *

pip = PipSQLPerformanceSchema()
output = pip.ask_pip("Tell me the total number of connections for each users.")
print(output)
print(pip.execute_query(output))

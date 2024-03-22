from pip_sql_performance_schema import *

pip = PipSQLPerformanceSchema()
output = pip.ask_pip("Which table has the highest number of inserts?")
print(output)
print(pip.execute_query(output))

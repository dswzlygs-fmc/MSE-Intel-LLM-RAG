from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
import uvicorn
# 假设你有一个名为ollama_model的模型，并且有一个名为predict的方法
# from your_ollama_module import ollama_model

from langchain.llms.ollama import Ollama
from langchain_community.graphs import Neo4jGraph

from langchain.chains.graph_qa.cypher import GraphCypherQAChain
# 定义一个工具包，包含执行Cypher查询的方法
import json

graph= Neo4jGraph(
    url="neo4j://127.0.0.1:7687",
    username="neo4j",
    password="dswzlygs",
    database="demo")


class MyCustomToolkit:
    def __init__(self, graph: Neo4jGraph):
        self._driver = graph

    def execute_cypher_query(self, cypher_query:str = None, **params):
        # 执行Cypher查询并返回结果
        try:
            return json.dumps(self._driver.query(cypher_query),indent=4)
        except Exception as e:
            print(f"An error occurred: {e}")
        

# toolkit = MyCustomToolkit(graph)

# # 定义Cypher查询
# cypher_query = "MATCH (n:Patient) RETURN n LIMIT 25"

# # 调用函数并传入toolkit的Cypher查询
# print(toolkit.execute_cypher_query(cypher_query))
# ...

cypher_generation_template_en = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything other than
for you to construct a Cypher statement. Do not include any text except
the generated Cypher statement. Make sure the direction of the relationship is
correct in your queries. Make sure you alias both entities and relationships
properly. Do not run any queries that would add to or delete from
the database. Make sure to alias all statements that follow as with
statement (e.g. WITH v as visit, c.billing_amount as billing_amount)
If you need to divide numbers, make sure to
filter the denominator to be non zero.

Examples:
# Who is the oldest patient and how old are they?
MATCH (p:Patient)
RETURN p.name AS oldest_patient,
       duration.between(date(p.dob), date()).years AS age
ORDER BY age DESC
LIMIT 1

# Which physician has billed the least to Cigna
MATCH (p:Payer)<-[c:COVERED_BY]-(v:Visit)-[t:TREATS]-(phy:Physician)
WHERE p.name = 'Cigna'
RETURN phy.name AS physician_name, SUM(c.billing_amount) AS total_billed
ORDER BY total_billed
LIMIT 1

# Which state had the largest percent increase in Cigna visits
# from 2022 to 2023?
MATCH (h:Hospital)<-[:AT]-(v:Visit)-[:COVERED_BY]->(p:Payer)
WHERE p.name = 'Cigna' AND v.admission_date >= '2022-01-01' AND
v.admission_date < '2024-01-01'
WITH h.state_name AS state, COUNT(v) AS visit_count,
     SUM(CASE WHEN v.admission_date >= '2022-01-01' AND
     v.admission_date < '2023-01-01' THEN 1 ELSE 0 END) AS count_2022,
     SUM(CASE WHEN v.admission_date >= '2023-01-01' AND
     v.admission_date < '2024-01-01' THEN 1 ELSE 0 END) AS count_2023
WITH state, visit_count, count_2022, count_2023,
     (toFloat(count_2023) - toFloat(count_2022)) / toFloat(count_2022) * 100
     AS percent_increase
RETURN state, percent_increase
ORDER BY percent_increase DESC
LIMIT 1

# How many non-emergency patients in North Carolina have written reviews?
MATCH (r:Review)<-[:WRITES]-(v:Visit)-[:AT]->(h:Hospital)
WHERE h.state_name = 'NC' and v.admission_type <> 'Emergency'
RETURN count(*)

String category values:
Test results are one of: 'Inconclusive', 'Normal', 'Abnormal'
Visit statuses are one of: 'OPEN', 'DISCHARGED'
Admission Types are one of: 'Elective', 'Emergency', 'Urgent'
Payer names are one of: 'Cigna', 'Blue Cross', 'UnitedHealthcare', 'Medicare',
'Aetna'

A visit is considered open if its status is 'OPEN' and the discharge date is
missing.
Use abbreviations when
filtering on hospital states (e.g. "Texas" is "TX",
"Colorado" is "CO", "North Carolina" is "NC",
"Florida" is "FL", "Georgia" is "GA", etc.)

Make sure to use IS NULL or IS NOT NULL when analyzing missing properties.
Never return embedding properties in your queries. You must never include the
statement "GROUP BY" in your query. Make sure to alias all statements that
follow as with statement (e.g. WITH v as visit, c.billing_amount as
billing_amount)
If you need to divide numbers, make sure to filter the denominator to be non
zero.

The question is:
{question}
"""

cypher_generation_template = """
任务：
为 Neo4j 图形数据库生成 Cypher 查询。

说明:
只使用模式中提供的关联类型和属性。
不要使用任何未提供的其他关联类型或属性。

Schema:
{schema}

注意:
在你的回答中不要包含任何解释或道歉。
不要回答任何除了要求你构建 Cypher 语句之外的问题。
除了生成的 Cypher 语句之外，不要包含任何文本。
确保你的查询中关联的方向是正确的。
确保你正确地为实体和关联设置了别名。
不要运行任何会增加或删除数据库内容的查询。
确保为所有跟随的语句使用别名，如 with 语句（例如 WITH v as visit, c.billing_amount as billing_amount）
如果你需要除以数字，请确保过滤分母为非零。

举例:
# "TX"州有哪些医院？
MATCH (n:Hospital) 
where n.hospital_state='TX'
RETURN n.hospital_name AS hospital_name 

# 名字是"Christina Williams"的患者的血型是什么？
MATCH (p:Patient {name: '‘'Christina Williams'})
return p.blood_type

# 名叫"Christina Williams"的患者就诊过几次？
MATCH (p:Patient {name: 'Christina Williams'})
RETURN count(p) AS patient_visits_count

# 谁是年龄最大的病人，他们的年龄是多少？
MATCH (p:Patient)
RETURN p.name AS oldest_patient,
       duration.between(date(p.dob), date()).years AS age
ORDER BY age DESC
LIMIT 1

# 哪位医生向 Cigna 收费最少？
MATCH (p:Payer)<-[c:COVERED_BY]-(v:Visit)-[t:TREATS]-(phy:Physician)
WHERE p.name = 'Cigna'
RETURN phy.name AS physician_name, SUM(c.billing_amount) AS total_billed
ORDER BY total_billed
LIMIT 1

# 从2022年到2023年，哪个州的 Cigna 访问量百分比增长最大？
MATCH (h:Hospital)<-[:AT]-(v:Visit)-[:COVERED_BY]->(p:Payer)
WHERE p.name = 'Cigna' AND v.admission_date >= '2022-01-01' AND
v.admission_date < '2024-01-01'
WITH h.state_name AS state, COUNT(v) AS visit_count,
     SUM(CASE WHEN v.admission_date >= '2022-01-01' AND
     v.admission_date < '2023-01-01' THEN 1 ELSE 0 END) AS count_2022,
     SUM(CASE WHEN v.admission_date >= '2023-01-01' AND
     v.admission_date < '2024-01-01' THEN 1 ELSE 0 END) AS count_2023
WITH state, visit_count, count_2022, count_2023,
     (toFloat(count_2023) - toFloat(count_2022)) / toFloat(count_2022) * 100
     AS percent_increase
RETURN state, percent_increase
ORDER BY percent_increase DESC
LIMIT 1

字符串类别值：
测试结果有以下几种：'Inconclusive', 'Normal', 'Abnormal'
访问状态有以下几种：'OPEN', 'DISCHARGED'
入院类型有以下几种：'Elective', 'Emergency', 'Urgent'
支付方名称有以下几种：'Cigna', 'Blue Cross', 'UnitedHealthcare', 'Medicare', 'Aetna'

如果访问的状态是“OPEN”并且出院日期缺失，则认为该访问是开放的。
在过滤医院所在州时使用缩写（例如，"Texas"是"TX"，"Colorado"是"CO"，"North Carolina"是"NC"，"Florida"是"FL"，"Georgia"是"GA"，等等）。

在分析缺失属性时，请确保使用 IS NULL 或 IS NOT NULL。
在你的查询中永远不要返回嵌入属性。你必须在查询中永远不要包含“GROUP BY”语句。
确保为所有跟随的语句使用别名，如 with 语句（例如 WITH v as visit, c.billing_amount as billing_amount）。
如果你需要除以数字，请确保过滤分母为非零。
尽可能少的使用关联关系，能用一个Node查询的内容，就不要关联多种Node

The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template_en
)

# ...

qa_generation_template_en = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a user's natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: [{context} , the provided information is empty ]

If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, assume the query
results are in units of days unless otherwise specified.

When names are provided in the query results, such as hospital names,
beware  of any names that have commas or other punctuation in them.
For instance, 'Jones, Brown and Murray' is a single hospital name,
not multiple hospitals. Make sure you return any list of names in
a way that isn't ambiguous and allows someone to tell what the full
names are.

Never say you don't have the right information if there is data in
the query results. Always use the data in the query results.

Helpful Answer:
"""

qa_generation_template = """你是一个助手，负责接收 Neo4j Cypher 查询的结果，并形成一个易于人类理解的中文回应。查询结果部分包含了基于用户自然语言问题生成的 Cypher 查询的结果。提供的信息是权威的，你绝不能怀疑它或尝试使用你的内部知识去纠正它。使答案听起来像是对问题的回答。

Query Results:
{context}

Question:
{question}

如果提供的信息为空，就说你不知道答案。
空信息看起来像这样：[]

如果提供的信息不为空，你必须使用结果提供答案。如果问题涉及时间持续期，除非另有规定，否则假定查询结果的单位是天。

当查询结果中提供了名字，如医院名称时，要注意任何包含逗号或其他标点的名字。例如，'Jones, Brown and Murray' 是一个单一的医院名称，而不是多个医院。确保你以一种不含糊且允许某人识别出完整名称的方式返回任何名称列表。

如果查询结果中有数据，永远不要说你没有正确的信息。始终使用查询结果中的数据。

有用的回答：
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)


llm = Ollama(model="qwen:7b",temperature=0.2)

graph= Neo4jGraph(
    url="neo4j://127.0.0.1:7687",
    username="neo4j",
    password="dswzlygs",
    database="demo")


hospital_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=llm,
    qa_llm=llm,
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,    
)

app = FastAPI()

# 定义请求模型
class InputData(BaseModel):
    data: str

# 创建一个简单的 API 端点
@app.post("/predict/")
async def predict(input_data: InputData):
    # 这里你将使用你的ollama模型进行预测
    try:
        # prediction = ollama_model.predict(input_data.data)
        prediction = hospital_cypher_chain.invoke(input_data.data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 运行 Uvicorn 服务器
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


# uvicorn main:app --reload
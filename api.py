import openai
import ast
from tree_sitter import Language, Parser

openai.api_base = 'https://api.openai-proxy.org/v1'
openai.api_key = 'sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU'

diff_prompt = "请根据给定的代码diff给出发生变化的函数，以python的list格式输出"
diff_file = open('diff/diff.cpp', 'r')
diff_context = diff_file.read()
diff_code = diff_context
response = openai.ChatCompletion.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "system", "content": diff_prompt},
        {"role": "user", "content": diff_code},
    ]
)
diff_list = ast.literal_eval(response.choices[0].message.content)
print(diff_list)

CPP_LANGUAGE = Language('build/my_language.so', 'cpp')

cpp_parser = Parser()
cpp_parser.set_language(CPP_LANGUAGE)
file = open('test/test.cpp', 'r')
context = file.read()
cpp_code = context
identifierlist = []
identifiertypes = ["identifier", "namespace_identifier", "qualified_identifier", "qualified_type_identifier",
                   "type_identifier", "field_identifier", "qualified_field_identifier",
                   "qualified_operator_cast_identifier", "dependent_field_identifier", "dependent_identifier",
                   "dependent_type_identifier"]
tree = cpp_parser.parse(bytes(cpp_code, "utf8"))
root_node = tree.root_node
cpp_code = cpp_code.split("\n")


def find_identifier(node, lst):
    if node.child_count != 0:
        for child in node.children:
            find_identifier(child, lst)
    else:
        line = node.start_point[0]
        start_point = node.start_point[1]
        end_point = node.end_point[1]
        if node.type in identifiertypes:
            identifier = cpp_code[line][start_point: end_point]
            lst.append([line, identifier])


find_identifier(root_node, identifierlist)
print(identifierlist)

relatedlines = []
for i in diff_list:
    for j in identifierlist:
        if i == j[1]:
            if j[0] not in relatedlines:
                relatedlines.append(j[0])

st = -1
for i in relatedlines:
    if st == -1:
        st = i
    elif st + 20 < i:
        st = i
    else:
        relatedlines.remove(i)
print(relatedlines)

motivation = []
length = len(cpp_code)
for line in relatedlines:
    code = cpp_code[line: min(line + 20, length)]
    code = "\n".join(code)
    motivation.append([line, min(line + 20, length), code])

print(motivation)
afterwards = []
for piece in motivation:
    main_prompt = "请根据给定的代码diff修改的代码块，注意，代码块有可能不需要被修改。请只返回修改后的代码块，不要返回任何除了代码之外的其他内容"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": main_prompt},
            {"role": "user", "content": "代码diff：\n" + diff_code + "\n代码块为：" + piece[2]},
        ]
    )
    ans = response.choices[0].message.content
    print(ans)
    afterwards.append([piece[0], piece[1], ans])



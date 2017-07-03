# 정규표현식
# Q1.a[.]{3,}b
a...b

# Q2
import re
p=re.compile('[a-z]+')
m=p.search("5 python")
print(m.start() + m.end())
# 10

# Q3
import re

data="""
park 010-9999-9999
kim 010-9909-7789
lii 010-8789-7768
"""
pat=re.compile("(?P<set>\d{3}[-]\d{4})[-](?P<replace>\d{4})") # 전화번호 형식 제공
result=pat.sub("\g<set>-####", data) # set 뒷부분 수정, 그룹명사용
print(result)

# Q4
import re
pat = re.compile(".*[@].*[.](?:com$|net$).*$")
print(pat.match("gskj@naver.com"))
<_sre.SRE_Match object; span=(0, 14), match='gskj@naver.com'>
print(pat.match("gskj@naver.co,kr"))
None

# XML 처리
# Q1
def indent(elem, level=0):
    i = "\n" + level*" "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i+ " "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

from xml.etree.ElementTree import Element, SubElement, dump

blog=Element("blog")
blog.attrib["date"] = "20151231"
subject=Element("subject")
subject.text="Why python?"
author=Element("author")
author.text="Eric"
blog.append(subject)
blog.append(author)
SubElement(blog, "content").text = "Life is too short, You need Python!"

indent(blog)
dump(blog)
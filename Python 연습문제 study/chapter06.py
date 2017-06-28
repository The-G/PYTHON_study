# Q1. 문자열 압축하기
input_number = input("문자열을 입력하세요!!!")
# print(input_number)
x = list(input_number)
# print(x)
count = 0
total_count = []
start_chr = []
for i in range(len(x)):
    if (i+1 == len(x)):
        count += 1
        total_count.append(count)
        start_chr.append(x[len(x) - 1])
    else:
        if (x[i] == x[i + 1]):
            count += 1
        else:
            count += 1
            total_count.append(count)
            count = 0
            start_chr.append(x[i])
print(start_chr)
print(total_count)
result_str = ''
for j in range(len(start_chr)):
        result_str += start_chr[j]
        result_str += str(total_count[j])
print(result_str)


# Q2. Duplicate Numbers
def check_num():
    raw_input_number = input("0~9 숫자를 입력해주세요, 띄어쓰기 가능!!")
    array_input_number = raw_input_number.split(" ")
    boolean_array = []
    for input_number in array_input_number:
        x = list(input_number)
        count = 0
        total_count = []
        start_chr = []
        for i in range(len(x)):
            if (i + 1 == len(x)):
                count += 1
                total_count.append(count)
                start_chr.append(x[len(x) - 1])
            else:
                if (x[i] == x[i + 1]):
                    count += 1
                else:
                    count += 1
                    total_count.append(count)
                    count = 0
                    start_chr.append(x[i])
        if (len(start_chr) != 10):
            boolean_array.append("false")
        else:
            for i in range(len(start_chr)):
                if(total_count[i] != 1):
                    boolean_array.append("false")
                    break
            boolean_array.append("true")
    return boolean_array

print(check_num())


# Q3. 모스 부호 해독
def decoding_mos(mos):
    dic = {'.-':'A', '-...':'B', '-.-.': 'C',
           '-..':'D', '.':'E', '..-.':'F',
           '--.':'G', '....':'H', '..':'I',
           '.---':'J', '-.-':'K', '.-..':'L',
           '--':'M', '-.':'N', '---':'O',
           '.--.':'P', '--.-':'Q', '.-.':'R',
           '...':'S', '-':'T', '..-':'U',
           '...-':'V', '.--':'W', '-..-':'X',
           '-.--':'Y', '--..':'Z'}
    if mos == None:
        mos = input("모스부호를 입력하세요!!")

    array_mos = mos.split(" ")
    result_text = ''
    # print(array_mos)
    for i in array_mos:
        try:
            result_text += dic[i]
        except:
            result_text += ' '
    return(result_text)

print(decoding_mos('.... .  ... .-.. . . .--. ...  . .- .-. .-.. -.--'))


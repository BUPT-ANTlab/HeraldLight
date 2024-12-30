try:
    with open("read.txt", "r", encoding="utf-8") as file:
        content = file.read()
    content = content.replace("\\n", "\n")
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(content)
    print("文件写入成功！")
except Exception as e:
    print("发生错误：", e)

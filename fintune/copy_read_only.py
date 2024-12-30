import os, shutil

shutil.copyfile('output1.json', 'output2.json')
print("文件已成功复制到 output2.json")

size1 = os.path.getsize('output1.json')
size2 = os.path.getsize('output2.json')
print(f"output1.json大小: {size1} 字节")
print(f"output2.json大小: {size2} 字节")

if size1 == size2:
    print("文件已完整复制。")
else:
    print("复制后的文件大小与源文件不同，请检查。")

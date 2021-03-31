import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

# directory = 'content/ibug'

# for filename in os.listdir(directory):
#     full_file_path = '';
#     if filename.endswith(".pts"): 
#         full_file_path = os.path.join(directory, filename)
#         print(full_file_path)
#         file1 = open(full_file_path, 'r')
#         Lines = file1.readlines()
#         count = 0

#         for line in Lines:
#             count += 1
#             # print("Line{}: {}".format(count, line.strip()))
#         print (count)


root = Element("data")
file1 = open('content/ibug/image_003_1.pts', 'r')

filename = SubElement(root, "filename")
filename.set("name", "image_003_1.pts")

Lines = file1.readlines()
count = 0

for line in Lines:
    count += 1
    if (count < 3):
        continue
    arr = line.strip().split()
    if (len(arr) > 1):
        x_val = SubElement(filename, 'x' + str(count-4))
        y_val = SubElement(filename, 'y' + str(count-4))
        [x,y] = arr
        x_val.text = x
        y_val.text = y
        # print("Line {}: {} {}".format(count-3, x, y))

xmlstr = minidom.parseString(tostring(root)).toprettyxml(indent="   ")
# xml_str = tostring(root).decode()
# print(text)
with open("New_Database.xml", "w") as f:
    f.write(xmlstr)
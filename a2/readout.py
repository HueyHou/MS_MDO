

data = '/Users/maelula/Documents/Python/207MDO/a2/tetrahedron_outputs/x.out'

wb = openpyxl.Workbook()
ws = wb.create_sheet("test")

ws.cell(row=x, column = y, value = data[x][y])
wb.save("/Users/maelula/Desktop")



import xlwt


# 创建一个excel工作表
xls = xlwt.Workbook()  
# 创建一个sheet, 并对sheet命名
sheet1 = xls.add_sheet('sheet1')


# 保存数据, 对excel命名
xls.save('test.xls')

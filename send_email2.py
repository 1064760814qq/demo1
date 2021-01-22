import yagmail

#发送的邮件中带内容，多张图片，多个文件（csv，xls，txt，doc，docx）

def sendmail():
    yag = yagmail.SMTP(user='1064760814@qq.com', password='snzyxqetuvzmbfjf', host='smtp.qq.com')
    #拿到你要发送的QQ邮箱的对象
    #很多情况下，你这个字符串的要用列表去存（方便多个字符串)
    #yagmail.inline()专门发送制定图片的,你指定图片必须要有路径！
    img_dir = 'C:/Users/10647/Desktop/图/微信图片_20201109101810.png'
    # img_dir2 = 'QQ图片20201108013136.png'

    contents = ['这是一些简单的复试流程与学习规划', yagmail.inline()]
    # '复试教学计划.docx'
    attachments = ['C:/Users/10647/Desktop/图/微信图片_20201109101810.png']
    persons = ['1139318027@qq.com', '2633611858@qq.com', '1250084010@qq.com', '642826959@qq.com']
    yag.send(to=persons, subject='测试发图片与文档邮件', contents=contents, attachments=attachments)

if __name__ == '__main__':
    sendmail()


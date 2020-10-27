import random
#oooooo
def ok():
    print('ok')
def choose_one():
    one_team = [1, 2, 3, 4, 5, 6]
    choose = random.choice(one_team)
    return choose

for i in range(8):
    print('第{}组'.format(i+1) +':{}'.format(choose_one()))




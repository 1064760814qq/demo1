import numpy as np
#BFS

class State:
    def __init__(self, state, parent=None):
        self.state = state
        # state is a ndarray with a shape(3,3) to storage the state
        self.parent = parent
        self.symbol = ' '


    def showInfo(self):
        for i in range(3):
            for j in range(3):
                print(self.state[i, j], end='   ')
            print("\n")
        print('->')
        return

    def getEmptyPos(self):
        postion = np.where(self.state == self.symbol)
        return postion

    def generateSubStates(self):
        subStates = []
        boarder = len(self.state) - 1
        # get the location
        row, col = self.getEmptyPos()
        if  col > 0:
        #it can move to left
            s = self.state.copy()
            temp = s.copy()
            s[row, col] = s[row, col-1]
            s[row, col-1] = temp[row, col]
            news = State(s,  parent=self)
            subStates.append(news)
        if  row > 0:
        #it can move to upper place
            s = self.state.copy()
            temp = s.copy()
            s[row, col] = s[row-1, col]
            s[row-1, col] = temp[row, col]
            news = State(s,  parent=self)
            subStates.append(news)
        if row < boarder:        #it can move to down place
            s = self.state.copy()
            temp = s.copy()
            s[row, col] = s[row+1, col]
            s[row+1, col] = temp[row, col]
            news = State(s, parent=self)
            subStates.append(news)
        if col < boarder:    #it can move to right place
            s = self.state.copy()
            temp = s.copy()
            s[row, col] = s[row, col+1]
            s[row, col+1] = temp[row, col]
            news = State(s,  parent=self)
            subStates.append(news)
        return subStates

    def solve(self):
        # open table
        openTable = []
       # close table
        closeTable = []
        # init
        openTable.append(self)
        # start loop
        steps = 1
        while len(openTable) > 0:
            n = openTable.pop(0)
            closeTable.append(n)
            subStates = n.generateSubStates()
            path = []

            for s in subStates:
                steps+=1
                if (s.state == s.answer).all():
                    while s.parent and s.parent != originState:
                        path.append(s.parent)
                        s = s.parent
                    path.reverse()
                    return path,steps
            openTable.extend(subStates)


if __name__ == '__main__':
    # the symbol representing the empty place
    # you can change the symbol at here
    symbolOfEmpty = ' '
    State.symbol = symbolOfEmpty
    # set the origin state of the puzzle
    originState = State(np.array([[2, 8, 3], [1, symbolOfEmpty, 4], [7, 6, 5]]))
    # and set the right answer in terms of the origin
    State.answer = np.array([[1, 2, 3], [8, State.symbol, 4], [7, 6, 5]])
    s1 = State(state=originState.state)
    path,steps = s1.solve()
    # steps = 0
    for node in path:
                # print the path from the origin to final state
                node.showInfo()
                # steps+=1
    print(State.answer)
    print("Total steps is %d" % steps)
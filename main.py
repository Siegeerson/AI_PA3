from nannon import *
import pickle
from scratch_nn import *
import random
from tqdm import tqdm


# Class for the value player that contains the loaded table to reduce on disk I/O time
class valuePlayerC:
    def __init__(self):
        self.table = pickle.load(open('nannon/mediocre_table.p', 'rb'))
    def valuePlayer(self,pos,roll):
        possiblePOS=legal_moves(pos,roll)
        best_move = pos
        bestScore = -1
        if possiblePOS[0]!=-1:
            # use python serializer to unpack the mediocre_table
            mediocre_table = self.table
            for possPOS in possiblePOS:
                tempPosition = make_move(pos,possPOS,roll)
                posVal = mediocre_table[tempPosition]
                if bestScore < posVal:
                    bestScore = posVal
                    best_move = tempPosition
            return best_move
        else:
            return pos

# player uses a provided nueral net
# to get the play function simply do somthing like
# netPlayerC(TD_Net.train_Net)).netPlayer
# will equal a function that plays using a TD net
class netPlayerC:
    def __init__(self,net):
        self.nn = net

    def netPlayer(self,pos,roll):
        possiblePOS=legal_moves(pos,roll)
        best_move = pos
        bestScore = -1
        if possiblePOS[0]!=-1:
            for possPOS in possiblePOS:
                tempPosition = make_move(pos,possPOS,roll)
                posVal = self.nn.query(tempPosition)[0][0]
                if bestScore < posVal:
                    bestScore = posVal
                    best_move = tempPosition
        return best_move


# moves based on which move presents your opponent with the least opportunity to improve
# a 1 deep min max tree
# possible score calculated with a NN
#pieces^2 * DICE
class mimMaxNNPlayerC:
    def __init__(self,net):
        self.nn=net

    def kbPlayer(self,pos,roll):
        possiblePOS=legal_moves(pos,roll)
        bestMove = -1
        bestMoveScore = -10
        oppScore = self.nn.query(swap_players(pos))
        for move in possiblePOS:
            # surface
            tempScore = 0
            situations =0
            tempBoard = swap_players(make_move(pos,move,roll))
            # for each roll
            for x in range(1,7):#assume sided die, easy to fix later with globals
                nMoves = legal_moves(tempBoard,x)
                moveTscore = 0
                # for each possible move
                for nMove in nMoves:
                    tTempBoard = swap_players(make_move(tempBoard,nMove,x))
                    situations += 1
                    moveTscore += self.nn.query(tTempBoard)[0][0] - self.nn.query(swap_players(tTempBoard))[0][0]
                tempScore = moveTscore/len(nMoves)
            tempScore = tempScore/6
            if tempScore > bestMoveScore:
                bestMoveScore = tempScore
                bestMove = move

        return make_move(pos,bestMove,roll)
# when the player wins it then goes and tells the network that all the moves it made are good
# to get a TD trained net just do TD_NET().train_Net() to return the nueral net
# is TD Gameon
class TD_NET:
    def __init__(self):
        self.nn = ScratchNetwork(3,6,1)
        self.trainNN = None
        self.moves = []
        self.start = False
        self.trainCount = 0

    def trainOnMoves(self,win):
        # impliment decay rate
        dec = .7 #decay rate
        trainVal = 5
        if not win:
            trainVal = -1
        for x in range(len(self.moves)):#trend towards winning positions
            finalTrainval= dec**(len(self.moves)-(x+1)) * trainVal
            self.nn.train(self.moves[x],finalTrainval)
        self.moves = []

    def TD_NetTrainPlay(self,pos,roll):
        if not self.start:
            self.start = True
            self.startPos = pos
        else:
            if self.startPos == pos:
                self.trainOnMoves(False)
        possiblePOS=legal_moves(pos,roll)
        best_move = pos
        bestScore = -1
        if possiblePOS[0]!=-1:
            for possPOS in possiblePOS:
                tempPosition = make_move(pos,possPOS,roll)
                posVal = self.nn.query(tempPosition)[0][0]
                if bestScore < posVal:
                    bestScore = posVal
                    best_move = tempPosition
                # if the game would end
                if who_won(tempPosition)==1:
                    # print("WIN")
                    self.trainOnMoves(True)
                    self.trainCount+=1
            self.moves.append(pos)
        return best_move

    def train_Net(self,n = 1):
        print("INIT LEARN")
        if not self.trainNN:
            self.trainNN = TD_NET() #setts up training partner
        for x in tqdm(range(n)):
            play_tourn(self.TD_NetTrainPlay,self.trainNN.TD_NetTrainPlay)
        return self.nn
#
#


a = TD_NET()
a.train_Net()
print(play_tourn(netPlayerC(a.nn).netPlayer,valuePlayerC().valuePlayer))
a.train_Net(4)
print(play_tourn(netPlayerC(a.nn).netPlayer,valuePlayerC().valuePlayer))

## NOTE:    Weights seem to be wierd and random
#           however values most likely stand in comparison to each other
#           rather than in comparison to values from table,
#           Still means they are hard to gauge
#input determines number of iterations of training shown, each training session trains in a 1000 round tourney 10 times
def testTD(n=1):
    pass
    net = TD_NET()
    table = pickle.load(open('nannon/mediocre_table.p', 'rb'))
    for x in range(n):
        net.train_Net(10)
        for pos in table.keys():
            print(pos)
            print(net.nn.query(pos)[0][0],((table[pos]-(-1))/2))
# testTD(3)




# hillclimb/mutation
















# # def basicNN(pos,roll):
# q = nn.query(target)
# print(q[0][0])
# print(scale(mediocre_table[target]))
# Target = ((2,7,7),(1,2,3))
# print(nn.query(Target)[0][0])
# print_board(Target)
# print(scale(mediocre_table[Target]))
# print(play_tourn(netPlayerC().netPlayer,rand_play))
# a = learningPlayerC()
# b = learningPlayerC()
# winP = []
# g= play_tourn(a.learningPlayer,valuePlayer,100)
# winP.append(g)
# print(g)
# for x in range(20):
#
#     play_tourn(a.learningPlayer,b.learningPlayer,10)
#     print("TOURN OVER")
#     g= play_tourn(a.learningPlayer,rand_play,1000)
#     winP.append(g)
#     print(g)
# with open('output.txt','w') as out:
#     for x in winP:
#         out.write(str(x))
#         out.write("\n")

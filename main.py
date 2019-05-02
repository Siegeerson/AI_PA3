from nannon import *
from scratch_nn import *
from hillClimb_NN import *
import pickle
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
class minMaxNNPlayerC:
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
        self.nn = ScratchNetwork(3,8,1)
        self.trainNN = None
        self.moves = []
        self.start = False
        self.trainCount = 0

    def trainOnMoves(self,win):
        # impliment decay rate
        dec = .5 #decay rate
        trainVal = -10
        if not win:
            trainVal = 10
        for x in range(len(self.moves)):#trend towards winning positions
            finalTrainval= dec**(len(self.moves)-(x+1)) * trainVal
            self.nn.train(self.moves[x],finalTrainval)
        self.moves = []

# TODO: Fix so the trianing gaurentees that it actually knows when it wins or loses
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
        # print("INIT LEARN")
        if not self.trainNN:
            self.trainNN = TD_NET() #setts up training partner
        for x in range(n):
            play_tourn(self.TD_NetTrainPlay,self.trainNN.TD_NetTrainPlay)#test training against value
        return self.nn
#
#

def testTDNET():
    a = TD_NET()
    for x in range(10):
        print(play_tourn(netPlayerC(a.nn).netPlayer,rand_play))
        a.train_Net(2)

## NOTE:    Weights seem to be wierd and random
#           however values most likely stand in comparison to each other
#           rather than in comparison to values from table,
#           Still means they are hard to gauge
#input determines number of iterations of training shown, each training session trains in a 1000 round tourney 10 times
def testTD(n=1):
    net = TD_NET()
    table = pickle.load(open('nannon/mediocre_table.p', 'rb'))
    for x in range(n):
        net.train_Net(10)
        for pos in table.keys():
            print(pos)
            print(net.nn.query(pos)[0][0],((table[pos]-(-1))/2))
# testTD(3)




# Matchbox
# NOTE: Steps
# 1 use explore function to build map
# 2 fill each box with 3 beads of each number 1-6
# 3 hash map of each position and each move chosen
# at win or loose go back through hash of moves chosen and either add or delete
class MenacePlayerC:
    def __init__(self):
        moveList = explore()
        checkerList =[]
        for x in range(3):
            for y in range(5):
                checkerList.append(x)
        self.boxCollection = dict()
        for state in moveList:
            self.boxCollection[state] = dict()
            # account for dice rolls
            for roll in range(1,7):
                self.boxCollection[state][roll] = checkerList[:]#copy
        self.moveCollection = dict()

    def MenacePlayer(self,pos,roll):
        checkerList = self.boxCollection[pos][roll][:]
        moveFound = False
        legalMoves = legal_moves(pos,roll)
        # condition for if no move possible
        if legalMoves[0] == -1:
            return make_move(pos,-1,roll)

        possibleMove = -1
        # get a legal move from the box
        while(not moveFound and len(checkerList)>0):
            possibleMove= random.choice(checkerList)
            if possibleMove in legalMoves:
                moveFound = True
            else:
                checkerList.remove(possibleMove)
                # should terminate
        if(not moveFound):
            possibleMove = random.choice(legalMoves)
        # add chosen move to list of moves taken
        moveList = []
        if not pos in self.moveCollection:
            self.moveCollection[pos] = dict()
        if  roll in self.moveCollection[pos]:
            moveList = self.moveCollection[pos][roll]
        moveList.append(possibleMove)
        self.moveCollection[pos][roll] = moveList
        return make_move(pos,possibleMove,roll)
    # public facing play method
    def publicMenacePlayer(self,pos,roll):
        checkerList = self.boxCollection[pos][roll][:]
        moveFound = False
        legalMoves = legal_moves(pos,roll)
        # condition for if no move possible
        if legalMoves[0] == -1:
            return make_move(pos,-1,roll)
        possibleMove = -1
        # get a legal move from the box
        while(not moveFound and len(checkerList)>0):
            possibleMove= random.choice(checkerList)
            if possibleMove in legalMoves:
                moveFound = True
            else:
                checkerList.remove(possibleMove)
                # should terminate
        if(not moveFound):
            possibleMove = random.choice(legalMoves)
        return make_move(pos,possibleMove,roll)

    def trainBox(self,games=10000,opp=valuePlayerC().valuePlayer):
        for x in range(games):
            winner = play_game(self.MenacePlayer,opp)
            if winner == "first":
                self.trainWin()
            else:
                self.trainLose()
    def trainWin(self):
        # get postions from game
        for pos in self.moveCollection:
            for roll in self.moveCollection[pos]:
                # get matching box for a position
                box = self.boxCollection[pos][roll]
                # add three of each choice made
                for x in range(2):
                    # for each choice add coresponing pebble to box
                    for move in self.moveCollection[pos][roll]:
                        box.append(move)
                # update box
                box.sort()
                self.boxCollection[pos][roll] = box
                # print(self.boxCollection[pos])
        # clear collection of moves
        self.moveCollection = dict()


    def trainLose(self):
        for pos in self.moveCollection:
            for roll in self.moveCollection[pos]:
                # get matching box for a position
                box = self.boxCollection[pos][roll]
                # for each choice add coresponing pebble to box
                for move in self.moveCollection[pos][roll]:
                    og = len(box)
                    # remove checker from box
                    if move in box:
                        box.remove(move)

                # update box
                self.boxCollection[pos][roll] = box
            # clear collection of moves
        self.moveCollection = dict()

# example of how box nn works
# not viable at low repetitions about the same as random
# most likely bad since it does not (currently) accuount for dice rolls just positions
# a better player could possibly be made by nesting
# UPDATE: after nesting dicts did not get better (when training for 100000 games)
# interesting that it does seem to teach itself what moves are impossible given a situation
def exampleOfBoxNN():
    d = MenacePlayerC()
    for x in range (10):
        d.trainBox(10000)
        print(play_tourn(d.publicMenacePlayer,rand_play))
    # print(play_tourn(d.publicMenacePlayer,rand_play,1000))
    with open('boxes.txt','w') as out:
        for x in d.boxCollection:
            out.write(str(x))
            out.write("\n")
            for y in d.boxCollection[x]:
                out.write(str(y))
                out.write("\n")
                out.write(str(d.boxCollection[x][y]))
                out.write("\n")
# exampleOfBoxNN()



# hill climber
# NOTE: Steps
# 1: set up empty net
# 2: set up mutated net
# 3: set up mating between them
class hillClimbC:
    def __init__(self):
        self.nn = hillClimb(3,5,1)
        print(self.nn.weights_ho)

    def train(self,nGens = 1000):
        climbHist=[]
        for x in tqdm(range(nGens)):
            mutatedNet = self.nn.getMutation()
            # print(self.nn.weights_ho,mutatedNet.weights_ho)
            mutWin = play_tourn(netPlayerC(mutatedNet).netPlayer,netPlayerC(self.nn).netPlayer,500,100)
            if mutWin > .5:
                self.nn = mutatedNet
                print("CHANGE",mutWin)
            climbHist.append(play_tourn(netPlayerC(self.nn).netPlayer,rand_play,500,100))

        return climbHist

def testClimb()
    hc = hillClimbC()
    climbH = hc.train(30)
    print(climbH)

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

#!/bin/python3.6

import chess
import chess.engine
import os
import random
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os.path



board = chess.Board()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 120)
        self.fc4 = nn.Linear(120, 50)
        self.fc5 = nn.Linear(50, 10)
        self.fc6 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x


net = Net()
game = 0
if(os.path.exists("model1.pt")):
   checkpoint = torch.load("model1.pt") 
   net.load_state_dict(checkpoint['net'])
   game = checkpoint['game']

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def parseFen(white):
    
    fen = board.fen().split(" ")[0].split("/")
    #print(fen)
    for i in range(len(fen)):
        fen[i] = fen[i].replace("2", "11") 
        fen[i] = fen[i].replace("3", "111") 
        fen[i] = fen[i].replace("4", "1111") 
        fen[i] = fen[i].replace("5", "11111") 
        fen[i] = fen[i].replace("6", "111111") 
        fen[i] = fen[i].replace("7", "1111111") 
        fen[i] = fen[i].replace("8", "11111111") 

    #print(fen)

    if(white):
        pieceChars = 'RNBQKBNRPPPPPPPPrnbqkbnrpppppppp'
    else:
        pieceChars = 'rnbqkbnrppppppppRNBQKBNRPPPPPPPP'
    pieces = []
    for i in pieceChars: #print(i)
        parsedFen = parseFenPiece(i, fen, white)
        pieces.append(parsedFen[0])
        pieces.append(parsedFen[1])
        pieces.append(parsedFen[2])
        fen = parsedFen[3]
        
    
    #print(pieces)

    return pieces

def parseFenPiece(piece, fen, white):
    #print(fen)
    #print("piece", piece)
    for y, row in enumerate(fen):
        for x, currentPiece in enumerate(row):
            if(piece == currentPiece):
                fen[y] = fen[y][:x] + "1" + fen[y][x+1:] 
                #print(x, y)
                if(white):
                    return (x/7, y/7, 1, fen)
                else:
                    return (x/7, 1-y/7, 1, fen)

    return (0, 0, 0, fen)

def parseMove(move):
            letterDict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
            x1 = letterDict[move.uci()[0]]/7
            y1 = (int(move.uci()[1])-1)/7
            x2 = letterDict[move.uci()[2]]/7
            y2 = (int(move.uci()[3])-1)/7
            return (x1, y1, x2, y2)


def pickMove(white):
        allBoardInput = []

        moves = list(board.legal_moves)
        for move in moves:
            boardInput = parseFen(white)
            #print(move.uci())
            x1, y1, x2, y2 = parseMove(move)
            boardInput.append(x1)
            boardInput.append(y1)
            boardInput.append(x2)
            boardInput.append(y2)
            #print(len(boardInput))
            allBoardInput.append(boardInput)
            #print(x1, y1, x2, y2)

        outputs = net(torch.Tensor(allBoardInput))
        _, choiceIndex = outputs.max(0)
        print(outputs)
        return (moves[choiceIndex], allBoardInput[choiceIndex])

def train(whiteX, blackX, whiteWins):



    running_loss = 0.0
    i = 0
    for epoch in range(2):  # loop over the dataset multiple times

        for inputs in whiteX:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print("white", outputs)
            loss = (outputs - torch.Tensor([whiteWins])).pow(2)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            i+=1


        for inputs in blackX:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print("black", outputs)
            loss = (outputs - torch.Tensor([not whiteWins])).pow(2)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            i+=1

    print("loss", running_loss/1)



while True:

    board = chess.Board()

    whiteX = []
    blackX = []
    whiteWins = False

    engineList = os.listdir("engines")
    engineName = "engines/"+engineList[random.randrange(len(engineList))]
    engine = chess.engine.SimpleEngine.popen_uci(engineName)
    #print(engineName)

    netWhite = random.randrange(2)

    board = chess.Board()

    while True:
        #print(board)

        if(netWhite):
            move, networkInput = pickMove(True)
            board.push(move)
            whiteX.append(networkInput)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            x1, y1, x2, y2 = parseMove(result.move)
            blackFen = parseFen(False)
            blackFen.append(x1)
            blackFen.append(y1)
            blackFen.append(x2)
            blackFen.append(y2)
            blackX.append(blackFen)
            board.push(result.move)


        #print(board)

        if(board.is_checkmate()):
            print("Game " + str(game) + ": White Wins")
            whiteWins = True
            engine.quit()
            break
        elif(board.is_game_over()):
            print("Game " + str(game) + ": It's a tie")
            engine.quit()
            break


        if(not netWhite):
            move, networkInput = pickMove(True)
            board.push(move)
            whiteX.append(networkInput)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            x1, y1, x2, y2 = parseMove(result.move)
            blackFen = parseFen(False)
            blackFen.append(x1)
            blackFen.append(y1)
            blackFen.append(x2)
            blackFen.append(y2)
            blackX.append(blackFen)
            board.push(result.move)


        if(board.is_checkmate()):
            print("Game " + str(game) + ": Black Wins")
            engine.quit()
            break
        elif(board.is_game_over()):
            print("Game " + str(game) + ": It's a tie")
            engine.quit()
            break

    
    train(torch.Tensor(whiteX), torch.Tensor(blackX), whiteWins)
    torch.save({
        'game': game,
        'net': net.state_dict()
        }, "model1.pt")


    


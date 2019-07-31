#!/bin/python3.6

import chess
import graphics as g
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F



board = chess.Board()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

net = Net()


def parseFen():
    
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

    pieceChars = 'rnbqkbnrppppppppRNBQKBNRPPPPPPPP'
    pieces = []
    for i in pieceChars:
        #print(i)
        parsedFen = parseFenPiece(i, fen)
        pieces.append(parsedFen[0])
        pieces.append(parsedFen[1])
        pieces.append(parsedFen[2])
        fen = parsedFen[3]
        
    
    #print(pieces)

    return pieces

def parseFenPiece(piece, fen):
    #print(fen)
    #print("piece", piece)
    for y, row in enumerate(fen):
        for x, currentPiece in enumerate(row):
            if(piece == currentPiece):
                fen[y] = fen[y][:x] + "1" + fen[y][x+1:] 
                #print(x, y)
                return (x/7, y/7, 1, fen)
    return (0, 0, 0, fen)

win = g.GraphWin("Cool Chess Game", 1000, 1000)
win.setBackground(g.color_rgb(0, 0, 0))

def render():
    boardImage = g.Image(g.Point(500, 500), "images/chessBoard.png")
    boardImage.draw(win)

    boardFen = parseFen()

    #blackRook = g.Image(g.Point(boardFen[0]*700+95, boardFen[1]*700+95), "images/blackRook.png")
    #blackKnight = g.Image(g.Point(boardFen[3]*700+95, boardFen[4]*700+95), "images/blackKnight.png")
    #print(boardFen[84]*7, boardFen[85]*7)


    renderPieces = []
    renderPieces.append((g.Image(g.Point(boardFen[0]*700+145, boardFen[1]*700+145), "images/blackRook.png"), boardFen[2]))
    renderPieces.append((g.Image(g.Point(boardFen[3]*700+145, boardFen[4]*700+145), "images/blackKnight.png"), boardFen[5]))
    renderPieces.append((g.Image(g.Point(boardFen[6]*700+145, boardFen[7]*700+145), "images/blackBishop.png"), boardFen[8]))
    renderPieces.append((g.Image(g.Point(boardFen[9]*700+145, boardFen[10]*700+145), "images/blackQueen.png"), boardFen[11]))
    renderPieces.append((g.Image(g.Point(boardFen[12]*700+145, boardFen[13]*700+145), "images/blackKing.png"), boardFen[14]))
    renderPieces.append((g.Image(g.Point(boardFen[15]*700+145, boardFen[16]*700+145), "images/blackBishop.png"), boardFen[17]))
    renderPieces.append((g.Image(g.Point(boardFen[18]*700+145, boardFen[19]*700+145), "images/blackKnight.png"), boardFen[20]))
    renderPieces.append((g.Image(g.Point(boardFen[21]*700+145, boardFen[22]*700+145), "images/blackRook.png"), boardFen[23]))
    renderPieces.append((g.Image(g.Point(boardFen[24]*700+145, boardFen[25]*700+145), "images/blackPawn.png"), boardFen[26]))
    renderPieces.append((g.Image(g.Point(boardFen[27]*700+145, boardFen[28]*700+145), "images/blackPawn.png"), boardFen[29]))
    renderPieces.append((g.Image(g.Point(boardFen[30]*700+145, boardFen[31]*700+145), "images/blackPawn.png"), boardFen[32]))
    renderPieces.append((g.Image(g.Point(boardFen[33]*700+145, boardFen[34]*700+145), "images/blackPawn.png"), boardFen[35]))
    renderPieces.append((g.Image(g.Point(boardFen[36]*700+145, boardFen[37]*700+145), "images/blackPawn.png"), boardFen[38]))
    renderPieces.append((g.Image(g.Point(boardFen[39]*700+145, boardFen[40]*700+145), "images/blackPawn.png"), boardFen[41]))
    renderPieces.append((g.Image(g.Point(boardFen[42]*700+145, boardFen[43]*700+145), "images/blackPawn.png"), boardFen[44]))
    renderPieces.append((g.Image(g.Point(boardFen[45]*700+145, boardFen[46]*700+145), "images/blackPawn.png"), boardFen[47]))

    renderPieces.append((g.Image(g.Point(boardFen[48]*700+145, boardFen[49]*700+145), "images/whiteRook.png"), boardFen[50]))
    renderPieces.append((g.Image(g.Point(boardFen[51]*700+145, boardFen[52]*700+145), "images/whiteKnight.png"), boardFen[53]))
    renderPieces.append((g.Image(g.Point(boardFen[54]*700+145, boardFen[55]*700+145), "images/whiteBishop.png"), boardFen[56]))
    renderPieces.append((g.Image(g.Point(boardFen[57]*700+145, boardFen[58]*700+145), "images/whiteQueen.png"), boardFen[59]))
    renderPieces.append((g.Image(g.Point(boardFen[60]*700+145, boardFen[61]*700+145), "images/whiteKing.png"), boardFen[62]))
    renderPieces.append((g.Image(g.Point(boardFen[63]*700+145, boardFen[64]*700+145), "images/whiteBishop.png"), boardFen[65]))
    renderPieces.append((g.Image(g.Point(boardFen[66]*700+145, boardFen[67]*700+145), "images/whiteKnight.png"), boardFen[68]))
    renderPieces.append((g.Image(g.Point(boardFen[69]*700+145, boardFen[70]*700+145), "images/whiteRook.png"), boardFen[71]))
    renderPieces.append((g.Image(g.Point(boardFen[72]*700+145, boardFen[73]*700+145), "images/whitePawn.png"), boardFen[74]))
    renderPieces.append((g.Image(g.Point(boardFen[75]*700+145, boardFen[76]*700+145), "images/whitePawn.png"), boardFen[77]))
    renderPieces.append((g.Image(g.Point(boardFen[78]*700+145, boardFen[79]*700+145), "images/whitePawn.png"), boardFen[80]))
    renderPieces.append((g.Image(g.Point(boardFen[81]*700+145, boardFen[82]*700+145), "images/whitePawn.png"), boardFen[83]))
    renderPieces.append((g.Image(g.Point(boardFen[84]*700+145, boardFen[85]*700+145), "images/whitePawn.png"), boardFen[86]))
    renderPieces.append((g.Image(g.Point(boardFen[87]*700+145, boardFen[88]*700+145), "images/whitePawn.png"), boardFen[89]))
    renderPieces.append((g.Image(g.Point(boardFen[90]*700+145, boardFen[91]*700+145), "images/whitePawn.png"), boardFen[92]))
    renderPieces.append((g.Image(g.Point(boardFen[93]*700+145, boardFen[94]*700+145), "images/whitePawn.png"), boardFen[95]))


    for i in renderPieces:
        if(i[1]):
            i[0].draw(win)



while True:
    print(board)
    render()
    print()

    while True:
        otherMoveText = input("")
        otherMove = 0
        try:
            otherMove = chess.Move.from_uci(otherMoveText)
        except ValueError:
            print("Use correct syntax: ")
            continue
        if(otherMove in board.legal_moves):
            board.push(otherMove)
            break
        print("That is an illegal move: ")
    
    print(board)
    render()
    print()

    if(board.is_checkmate()):
        print("Winner winner chicken dinner")
    elif(board.is_game_over()):
        print("It's a tie")
    
    allBoardInput = []

    moves = list(board.legal_moves)
    for move in moves:
        boardInput = parseFen()
        #print(move.uci())
        letterDict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        x1 = letterDict[move.uci()[0]]/7
        y1 = (int(move.uci()[1])-1)/7
        x2 = letterDict[move.uci()[2]]/7
        y2 = (int(move.uci()[3])-1)/7
        boardInput.append(x1)
        boardInput.append(y1)
        boardInput.append(x2)
        boardInput.append(y2)
        #print(len(boardInput))
        allBoardInput.append(boardInput)
        #print(x1, y1, x2, y2)

    outputs = net(torch.Tensor(allBoardInput))
    _, choiceIndex = outputs.max(0)
    print(choiceIndex)

    board.push(moves[choiceIndex])

    if(board.is_checkmate()):
        print("You Lost")
    elif(board.is_game_over()):
        print("It's a tie")



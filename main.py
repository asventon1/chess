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
                fen[y] = fen[y][:x] + " " + fen[y][x+1:] 
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

    renderPieces = []
    renderPieces.append(g.Image(g.Point(boardFen[0]*700+145, boardFen[1]*700+145), "images/blackRook.png"))
    renderPieces.append(g.Image(g.Point(boardFen[3]*700+145, boardFen[4]*700+145), "images/blackKnight.png"))

    for i in renderPieces:
        i.draw(win)



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



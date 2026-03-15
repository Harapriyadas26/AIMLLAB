procedure Solve8Queens():
    board[8] ← empty
    placeQueen(0)

procedure placeQueen(row):
    if row == 8:
        print board   // solution found
        return
    for col from 0 to 7:
        if safe(row, col):
            board[row] ← col
            placeQueen(row + 1)
            board[row] ← empty  // backtrack

function safe(row, col):
    for i from 0 to row-1:
        if board[i] == col or diagonal conflict with (row,col):
            return false
    return true

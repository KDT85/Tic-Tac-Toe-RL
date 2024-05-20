import numpy as np
import time
import pickle
import pandas as pd
from tqdm import tqdm

EMPTY = '_'
GRID_SIZE = 3
PLAYER = ['X', 'O']

def show_board(board):
    board = np.array(board)
    print()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            print('|', end='')
            print(board[i, j], end='')
        print('|')
    print()

def get_legal_moves(board):
    legal_moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                legal_moves.append((i, j))
    return legal_moves

def get_human_move(player):
    human = input(f"Player {player}, enter a square: ")
    valid = map_index(human)
    while not valid:
        print("Invalid input! Please enter a number between 1 and 9.")
        return get_human_move(player)
    return valid[1]

def map_index(int_choice):
    num_pad_map = {'7': (0, 0), '8': (0, 1), '9': (0, 2), '4': (1, 0), '5': (1, 1), '6': (1, 2), '1': (2, 0), '2': (2, 1), '3': (2, 2)}
    if int_choice in num_pad_map:
        return True, num_pad_map[int_choice]
    else:
        return False

def make_move(move, board, player):
    row, col = move
    if board[row, col] == EMPTY:
        board[row, col] = player
    else:
        print("That square is already taken!")
        move = get_human_move(player)  # Ask for input again
        make_move(move, board, player)
    return board

def check_win(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != EMPTY:
            return True, row[0]

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != EMPTY:
            return True, board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != EMPTY:
        return True, board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != EMPTY:
        return True, board[0][2]

    # Check draw
    if all(board[i][j] != EMPTY for i in range(3) for j in range(3)):
        return True, 'Draw'
    return False, None

def state_to_index(state):
    mapping = {'X': 1, 'O': -1, EMPTY: 0}
    index = 0
    for i, value in enumerate(state):
        index += (3 ** i) * mapping[value]
    return abs(index)

def get_action(legal_moves, Q_table, state_index, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.choice(len(legal_moves))
        return legal_moves[action]
    else:
        best_action = None
        best_q_value = float('-inf')
        for action in legal_moves:
            action_index = action[0] * GRID_SIZE + action[1]
            q_value = Q_table[state_index][action_index]
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        return best_action

def opponent(moves):
    move = np.random.randint(len(moves))
    return moves[move]

def train_agent(episodes):
    q_table = np.zeros((3 ** (GRID_SIZE * GRID_SIZE), GRID_SIZE * GRID_SIZE))
    start_time = time.time()
    print("Training started...")

    for episode in tqdm(range(episodes), desc="Training Progress"):
        winner = False
        board = [[EMPTY] * GRID_SIZE for _ in range(GRID_SIZE)]
        turn = 0
        while not winner:
            legal_moves = get_legal_moves(board)
            state = tuple(np.array(board).flatten())
            state_index = state_to_index(state)

            if PLAYER[turn % 2] == 'X':
                action = get_action(legal_moves, q_table, state_index, epsilon)
                row, col = action
            else:
                row, col = opponent(legal_moves)

            board[row][col] = PLAYER[turn % 2]
            winner, result = check_win(board)

            if winner:
                if result == 'X':
                    reward = 1
                elif result == 'O':
                    reward = -1
                else:
                    reward = 0
            else:
                reward = 0

            next_state = tuple(np.array(board).flatten())
            next_state_index = state_to_index(next_state)
            action_index = row * GRID_SIZE + col
            q_table[state_index][action_index] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index][action_index])

            turn += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time} seconds")
    return q_table

def save_q_table(q_table, filename):
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)

def load_q_table(filename):
    with open(filename, 'rb') as f:
        q_table = pickle.load(f)
    return q_table

def save_q_table_to_excel(q_table, filename):
    df = pd.DataFrame(q_table)
    df.to_excel(filename, index=False)
    print(df)

def play_game(q_table, wins, losses, draws):
    board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=str)
    board.fill(EMPTY)
    game_over = False
    round = 0

    while not game_over:
        show_board(board)
        if PLAYER[round % 2] == 'X':
            state = tuple(np.array(board).flatten())
            state_index = state_to_index(state)
            move = get_action(get_legal_moves(board), q_table, state_index, 0)
            print(f"AI played at {move}")
        else:
            #move = get_human_move(PLAYER[round % 2])
            move = opponent(get_legal_moves(board))
        board = make_move(move, board, PLAYER[round % 2])
        game_over, result = check_win(board)
        round += 1

    show_board(board)
    if result == 'Draw':
        print("It's a draw!")
        draws += 1
    else:
        print(f"Player {result} wins!")
        if result == 'X':
            wins +=1
        else:
            losses +=1
    
    return (wins, losses, draws)

# Initialize Q-learning parameters
alpha = 0.3  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1  # Exploration rate
episodes = 20000000  # number of episodes for training

# Train the agent
#q_table = train_agent(episodes)

# Save the Q-table
#save_q_table(q_table, 'q_table.pkl')
#save_q_table_to_excel(q_table, f'q_table_{episodes}_episodes.xlsx')

# Load the Q-table (if needed)
q_table = load_q_table('q_table.pkl')

wins, losses, draws = 0, 0, 0
# Play against the trained AI
for i in range(1000):
    wins, losses, draws = play_game(q_table, wins, losses, draws)



# Create a DataFrame
df = pd.DataFrame({
    'Result': ['Wins', 'Losses', 'Draws'],
    'Count': [wins, losses, draws]
})

# Print the DataFrame
print(df.to_string(index=False))

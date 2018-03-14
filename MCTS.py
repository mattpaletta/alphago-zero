import math

root_node = None

#keeps track of whose turn it is
friendly_turn = -1

#linked list style
class Node:
   def __init__(self, parent, board, turn):
      #variable for determining whose turn the board position is from
      self.friendly_turn = turn

      self.parent = parent
      self.board = board

      #will always be between -1 and 1. the value passed back to the network will be divided by N
      #will be either -1 or 1 depending on if its a terminal state
      #naturally from point of view of network. multiply by friendly_turn to get whose turn it is
      self.W = determine_if_terminal(board)
      if(self.W==0):
         self.W = return_value_from_network()
      self.N = 0     #visit count 

      
      self.P = return_value_from_network() #probability of being chosen
      self.C = 1     #exploration constant
      self.children = [None]
   def calculateU(self):
      self.N+=1

      #U is score for how valuable this position is in being determined
      self.U = self.W*self.friendly_turn/self.N+self.C*self.P*math.sqrt(math.log(self.parent.N)/(1+self.N))

      #propogate W back up network
      self.parent.W += self.W    
      push_value_to_network("W", self.board, self.W/self.N)


#recursively selects hghest value child at each level
def select(node):
   friendly_turn*=-1       #alternates turns going down
   if len(node.children) != 0:
      max_u = 0
      max_node = None
      for x in range(len(node.children)):
         if(node.children[x].U > max_u):
            max_u = node.children[x].U
            max_node = node.children[x]
      return select(max_node)
   #if it doesn't have children kill the recursion
   else:
      return node


#creates all possible children from current state
#if this isnt a terminal board position
def expand(node, board_layout):
   if(determine_if_terminal(board_layout) == 0):
      #for each board reachable by the current board
      possible_outcomes = return_value_from_network() #array of adjacent board positions
      for x in range(len(possible_outcomes)):
         child = Node(node, possible_outcomes[x], friendly_turn)
         node.children.append(child)
         update(child)


#propogates new values up the network
def update(node):
   if(node != root_node):
      node.calculateU()
      update(node.parent)

#returns -1 if lost from position, 1 if won. o otherwise
def determine_if_terminal(board_layout):
   return False

def push_value_to_network(type, node, value):
   return 1

def return_value_from_network():     #replace this with values from network
   return 1


#the function. assumes it always starts out on friendly turn
def find_optimal_path(board_layout):
   root_node = Node(None, board_layout, -1)
   while(searching):
      friendly_turn = -1
      temp_node = select(root_node)
      expand(temp_node, temp_node.board)
   most_explore_count = 0
   most_explored = None
   for x in range(len(root_node.children)):
      if(root_node.children[x].N > most_explore_count):
         most_explore_count = root_node.children[x].N
         most_explored = root_node.children[x]
   for x in range(len(root_node.children)):
      if(root_node.children[x]==most_explored):
         push_value_to_network("P", root_node.children[x].board, 1)
      else:
         push_value_to_network("P", root_node.children[x].board, 0)
   return most_explored.board    #could return position of next move instead?
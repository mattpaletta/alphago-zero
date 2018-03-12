import math

root_node = None
friendly_turn = 1

class Node:
   
   def __init__(self, parent, board):
      self.parent = parent
      self.board = board      #2

      #will always be between -1 and 1. the value passed back to the network will be divided by N
      #will be either -1 or 1 depending on if its a terminal state
      #naturally from point of view of network. multiply by friendly_turn to get whose turn it is
      self.W = return_value_from_network() #get W from network     
      self.N = 0
      self.P = return_value_from_network() #get p from network
      self.C = 1     #exploration constant
      self.children = [None]
   def calculateU(self):      #TODO will also have to take into account whose turn it is
      self.U = self.W*friendly_turn/self.N+self.C*self.P*math.sqrt(math.log(self.parent.N)/(1+self.N))
      self.N+=1
      self.parent.W += self.W    
      push_value_to_network("W", self, self.W/self.N)    #TODO update the W NN with this value. 


#to start, insert root node
#will need to know whose turn it is. W will will just be inverted if opponent
def select(node):
   if len(node.children) != 0:
      friendly_turn*=-1       #alternates turns going down
      #recursively select best child
      max_u = 0
      max_node = None
      for x in range(len(node.children)):
         if(node.children[x].U > max_u):
            max_u = node.children[x].U
            max_node = node.children[x]
      return max_node
   #if it doesn't have children kill the recursion
   else:
      return node


def expand(node):
   #for each board reachable by the current board
   possible_outcomes = return_value_from_network() #array of adjacent board positions
   for x in range(len(possible_outcomes)):
      child = Node(node, possible_outcomes[x])
      node.children.append(child)
      update(child)


def update(node):
   if(node != root_node):
      node.calculateU()
      friendly_turn*=-1	         #alternates turns coming back up
      update(node.parent)

def push_value_to_network(type, node, value):
   return 0

def return_value_from_network():     #replace this with values from network
   return 1


#the function. assumes it always starts out on friendly turn
def find_optimal_path(board_layout):
   root_node = Node(None, board_layout)
   while(searching):
      friendly_turn = 1
      temp_node = select(root_node)
      expand(temp_node)
   most_explore_count = 0
   most_explored = None
   for x in range(len(root_node.children)):
      if(root_node.children[x].N > most_explore_count):
         most_explore_count = root_node.children[x].N
         most_explored = root_node.children[x]
   for x in range(len(root_node.children)):
      if(root_node.children[x]==most_explored):
         push_value_to_network("P", root_node.children[x], 1)
      else:
         push_value_to_network("P", root_node.children[x], 0)
   return most_explored
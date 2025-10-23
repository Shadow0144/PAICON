# -*- coding: utf-8 -*-
import random as rand
import numpy as np
from enum import Enum
import copy

# Next action equation:
# a_(t + 1) = a_t - k * gradient_g(a_t)
# k is a scaling factor

# gradient*_g(a_t) = argmax(gradient_g(a_t) for all G(g, a_t), ||gradient_g(a_t)||)

# Odds that X is uncovered / clear:
# c_t(X) = 1 - (1 / ||B||) * sum(o_t(Y, X) for all Y in B)
# Odds that Y is on top of X:
# o_t(X, Y) = o_(t - 1)(X, Y) + c_(t - 1)(X) * c_(t - 1)(Y) * a_stack(X, Y) - c_(t - 1) * a_unstack(X, Y)

# Gradients:
# gradient_stack = (partial_g / partial_o(X, Y)) * (partial_o(X, Y) / partial_a_stack(X, Y))
# gradient_unstack = (partial_g / partial_o(X, Y)) * (partial_o(X, Y) / partial_c(X)) * 
#                      (partial_c(X) / partial_o(Z, X)) * (partial_o(Z, X) / partial_a_unstack(Z, X))

class COLOR(Enum):
    BLACK = 0
    GREY = 1
    WHITE = 2
    RED = 3
    VERMILION = 4
    ORANGE = 5
    AMBER = 6
    YELLOW = 7
    CHARTEUSE = 8
    GREEN = 9
    TEAL = 10
    BLUE = 11
    VIOLET = 12
    PURPLE = 13
    MAGENTA = 14
    
max_color = 14

no_plan_solver = None

class Block:
    def __init__(self, index, row, col, color):
        self.index = index
        self.row = row
        self.col = col
        self.color = color
        self.block_above = None
        self.block_below = None

class BlocksWorld:
    def __init__(self, num_blocks):
        if num_blocks > max_color:
            raise ValueError("Not enough colors")
        
        self.num_blocks = num_blocks
            
        self.colors = rand.sample(range(max_color + 1), num_blocks)
        self.blocks = np.empty(num_blocks, dtype=Block)
        
        self.init_world = np.empty((num_blocks, num_blocks), dtype=Block)
        for b in range(0, num_blocks):
            col = rand.randint(0, num_blocks - 1)
            row = 0
            block_below = None
            while self.init_world[row, col] is not None:
                block_below = self.init_world[row, col]
                row = row + 1
                # It should not be possible to build a column bigger than the number of 
                # available columns so no check needed
            self.blocks[b] = Block(b, row, col, self.colors[b])
            self.blocks[b].block_below = block_below
            if block_below is not None:
                block_below.block_above = self.blocks[b]
            self.init_world[row, col] = self.blocks[b]
        self.current_world = copy.copy(self.init_world)
        
    def can_stack(self, src_block, dst_block):
        return (src_block != dst_block and 
                src_block.block_above is None and
                dst_block.block_above is None)
    
    def can_unstack(self, src_block):
        return (src_block.block_above is None)
            
    def stack(self, src_block, dst_block):
        if not self.can_stack(src_block, dst_block):
            raise RuntimeError("One or both blocks are not clear of blocks above them or are the same block")
            
        self.current_world[src_block.row, src_block.col] = None
        self.current_world[dst_block.row + 1, dst_block.col] = src_block
        
        if src_block.block_below is not None:
            src_block.block_below.block_above = None
        src_block.block_below = dst_block
        dst_block.block_above = src_block
        
        src_block.row = dst_block.row + 1
        src_block.col = dst_block.col
        
    def unstack(self, src_block):
        if not self.can_unstack(src_block):
            raise RuntimeError("The block is not clear of blocks above it")
            
        self.current_world[src_block.row, src_block.col] = None
        for col in range(0, self.num_blocks):
            if self.current_world[0, col] is None:
                self.current_world[0, col] = src_block
                src_block.row = 0
                src_block.col = col
                break
        
        if src_block.block_below is not None:
            src_block.block_below.block_above = None
        src_block.block_below = None
        
    def print_full(self):
        print("+", end="")
        for col in range(1, self.num_blocks):
            print("----", end="")
        print("---+")
        for row in range(self.num_blocks - 1, -1, -1): # Print the rows backwards 
            for col in range(0, self.num_blocks):
                if self.current_world[row, col] is not None:
                    if self.current_world[row, col].index < 10:
                        print(f"| {self.current_world[row, col].index} ", end="")
                    else:
                        print(f"|{self.current_world[row, col].index} ", end="")
                else:
                    print("|   ", end="")
            print("|")
            if (row != 0):
                for col in range(1, self.num_blocks - 1):
                    print("-----", end="")
                print("-", end="")
                print()
        print("+", end="")
        for row in range(1, self.num_blocks):
            print("----", end="")
        print("---+")
        
    def print(self):
        print("+", end="")
        for col in range(1, self.num_blocks):
            print("----", end="")
        print("---+")
        for row in range(self.num_blocks - 1, -1, -1): # Print the rows backwards
            # Check if the row is empty before printing it
            skip_row = True
            for col in range(0, self.num_blocks):
                if self.current_world[row, col] is not None:
                    skip_row = False
                    break
            if not skip_row:
                for col in range(0, self.num_blocks):
                    if self.current_world[row, col] is not None:
                        if self.current_world[row, col].index < 10:
                            print(f"| {self.current_world[row, col].index} ", end="")
                        else:
                            print(f"|{self.current_world[row, col].index} ", end="")
                    else:
                        print("|   ", end="")
                print("|")
                if (row != 0):
                    for col in range(1, self.num_blocks - 1):
                        print("-----", end="")
                    print("-", end="")
                    print()
        print("+", end="")
        for row in range(1, self.num_blocks):
            print("----", end="")
        print("---+")
        
def shuffle_world(blocks_world, max_steps):
    shuffled_world = copy.copy(blocks_world)
    for t in range(0, max_steps):
        action = rand.randint(0, 1)
        block_x = shuffled_world.blocks[rand.randint(0, shuffled_world.num_blocks - 1)]
        if action == 0:
            block_y = shuffled_world.blocks[rand.randint(0, shuffled_world.num_blocks - 1)]
            if shuffled_world.can_stack(block_x, block_y):
                shuffled_world.stack(block_x, block_y)
        else: # action == 1
            if shuffled_world.can_unstack(block_x):
                shuffled_world.unstack(block_x)
    return shuffled_world

class Goal:
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y
        
class NoPlanSolver:
    def __init__(self, blocks_world, goals=None):
        self.blocks_world = blocks_world
        self.goals = goals
        self.last_state = copy.copy(self.blocks_world)
        self.num_steps = 0
        
        # The odds we took an action the previous timestep (which is 0 for everything)
        self.a_stack = np.zeros((self.blocks_world.num_blocks, self.blocks_world.num_blocks), dtype=float)
        self.a_unstack = np.zeros((self.blocks_world.num_blocks, self.blocks_world.num_blocks), dtype=float)
        
        # Create a grid of initial estimates for all c_0(X) and o_0(X, Y)
        self.previous_c_t_est = np.zeros(self.blocks_world.num_blocks, dtype=float)
        self.previous_o_t_est = np.zeros((self.blocks_world.num_blocks, self.blocks_world.num_blocks), dtype=float)
        
        # A block is on top of another block if it is in the same column with a higher row number
        # (so it also includes those not directly above)
        for col in range(0, self.blocks_world.num_blocks):
            for row in range(0, self.blocks_world.num_blocks):
                current_block = self.blocks_world.current_world[row, col]
                if current_block is not None:
                    self.previous_c_t_est[current_block.index] = 1 # Temporarily assume it's clear
                    if row < self.blocks_world.num_blocks: # Check all the blocks above
                        row = row + 1
                        next_block = self.blocks_world.current_world[row, col]
                        if next_block is not None:
                            self.previous_o_t_est[next_block.index, current_block.index] = 1
                            self.previous_c_t_est[current_block.index] = 0 # Was not clear, so put back to 0
                        #else: 
                        #    break # If the block is empty, the rest of the column is empty
                else:
                     break # Stop checking this column if we hit an empty space
                     
        self.current_c_t_est = copy.copy(self.previous_c_t_est)
        self.current_o_t_est = copy.copy(self.previous_o_t_est)
            
    def get_c_t(self, X):
        # Odds X is clear of blocks on top
        ### c_t(X) = 1 - (1 / ||B||) * sum(o_t(Y, X) for all Y in B)
        # c_t(X) = 1 - sum(o_t(Y, X) for all Y in B)
        sum = 0.0
        for Y in range(0, self.blocks_world.num_blocks):
            sum = sum + self.get_o_t(Y, X)
        #sum = 1 - (sum / self.blocks_world.num_blocks)
        sum = 1 - sum
        return sum
        
    def get_o_t(self, X, Y):
        # Odds X is on top of Y
        # o_t(X, Y) = o_(t - 1)(X, Y) + c_(t - 1)(X) * c_(t - 1)(Y) * a_stack(X, Y) - c_(t - 1) * a_unstack(X, Y)
        sum = 0.0
        sum = sum + self.get_o_t_1(X, Y)
        sum = sum + self.get_c_t_1(X) * self.get_c_t_1(Y) * self.a_stack[X, Y]
        sum = sum - self.get_c_t_1(X) * self.a_unstack[X, Y]
        return sum
    
    def get_c_t_1(self, X):
        return self.previous_c_t_est[X]
    
    def get_o_t_1(self, X, Y):
        return self.previous_o_t_est[X, Y]
        
    def create_estimates(self):
        self.current_c_t_est = np.zeros(self.blocks_world.num_blocks, dtype=float)
        self.current_o_t_est = np.zeros((self.blocks_world.num_blocks, self.blocks_world.num_blocks), dtype=float)
        
        for x in range(0, self.blocks_world.num_blocks):
            self.current_c_t_est[x] = self.get_c_t(x)
            
        for x in range(0, self.blocks_world.num_blocks):
            for y in range(0, self.blocks_world.num_blocks):
                self.current_o_t_est[x, y] = self.get_o_t(x, y)
            
    def get_gradient_a_stack(self, X, Y):
        print("TODO")
        
    def get_gradient_a_unstack(self, X, Y):
            print("TODO")
    
    def next_time_step(self):
        self.create_estimates()
        self.previous_c_t_est = self.current_c_t_est
        self.previous_o_t_est = self.current_o_t_est
        
    def stack(self, src_block_idx, dst_block_idx):
        src_block = self.blocks_world.blocks[src_block_idx]
        dst_block = self.blocks_world.blocks[dst_block_idx]
        if self.blocks_world.can_stack(src_block, dst_block):
            self.a_stack = np.zeros((self.blocks_world.num_blocks, self.blocks_world.num_blocks), dtype=float)
            self.a_unstack = np.zeros((self.blocks_world.num_blocks, self.blocks_world.num_blocks), dtype=float)
            self.a_stack[src_block_idx, dst_block_idx] = 1
            self.blocks_world.stack(src_block, dst_block)
            self.next_time_step()
            
    def unstack(self, src_block_idx):
        src_block = self.blocks_world.blocks[src_block_idx]
        # There is no merit to unstacking if there's nothing below the block, so make sure there is something
        if self.blocks_world.can_unstack(src_block) and src_block.block_below is not None:
            self.a_stack = np.zeros((self.blocks_world.num_blocks, self.blocks_world.num_blocks), dtype=float)
            self.a_unstack = np.zeros((self.blocks_world.num_blocks, self.blocks_world.num_blocks), dtype=float)
            self.a_unstack[src_block_idx, src_block.block_below.index] = 1
            self.blocks_world.unstack(src_block)
            self.next_time_step()
        
    def perform_next_action(self):
        print("TODO")

def create_blocks_world(num_blocks):
    global blocks_world
    blocks_world = BlocksWorld(num_blocks)
    print("Initial world:")
    blocks_world.print()
    
    print()
    
    print("Shuffled world:")
    shuffled_world = shuffle_world(blocks_world, 20)
    shuffled_world.print()
    
    global no_plan_solver
    no_plan_solver = NoPlanSolver(shuffled_world)
    
if __name__ == "__main__":
    create_blocks_world(10)
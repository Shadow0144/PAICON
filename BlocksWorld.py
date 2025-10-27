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
        
        self.world = np.empty((num_blocks, num_blocks), dtype=Block)
        for b in range(0, num_blocks):
            col = rand.randint(0, num_blocks - 1)
            row = 0
            block_below = None
            while self.world[row, col] is not None:
                block_below = self.world[row, col]
                row = row + 1
                # It should not be possible to build a column bigger than the number of 
                # available columns so no check needed
            self.blocks[b] = Block(b, row, col, self.colors[b])
            self.blocks[b].block_below = block_below
            if block_below is not None:
                block_below.block_above = self.blocks[b]
            self.world[row, col] = self.blocks[b]
        
    def can_stack(self, src_block, dst_block):
        return (src_block != dst_block and 
                src_block.block_above is None and
                dst_block.block_above is None)
    
    def can_unstack(self, src_block):
        return (src_block.block_above is None)
            
    def stack(self, src_block, dst_block):
        if not self.can_stack(src_block, dst_block):
            raise RuntimeError("One or both blocks are not clear of blocks above them or are the same block")
            
        self.world[src_block.row, src_block.col] = None
        self.world[dst_block.row + 1, dst_block.col] = src_block
        
        if src_block.block_below is not None:
            src_block.block_below.block_above = None
        src_block.block_below = dst_block
        dst_block.block_above = src_block
        
        src_block.row = dst_block.row + 1
        src_block.col = dst_block.col
        
    def unstack(self, src_block):
        if not self.can_unstack(src_block):
            raise RuntimeError("The block is not clear of blocks above it")
            
        self.world[src_block.row, src_block.col] = None
        for col in range(0, self.num_blocks):
            if self.world[0, col] is None:
                self.world[0, col] = src_block
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
                if self.world[row, col] is not None:
                    if self.world[row, col].index < 10:
                        print(f"| {self.world[row, col].index} ", end="")
                    else:
                        print(f"|{self.world[row, col].index} ", end="")
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
                if self.world[row, col] is not None:
                    skip_row = False
                    break
            if not skip_row:
                for col in range(0, self.num_blocks):
                    if self.world[row, col] is not None:
                        if self.world[row, col].index < 10:
                            print(f"| {self.world[row, col].index} ", end="")
                        else:
                            print(f"|{self.world[row, col].index} ", end="")
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
    shuffled_world = copy.deepcopy(blocks_world)
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
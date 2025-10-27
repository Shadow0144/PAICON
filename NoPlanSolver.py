# -*- coding: utf-8 -*-
import BlocksWorld

import random as rand
import numpy as np
import copy
        
class NoPlanSolver:
    class Goal:
        def __init__(self, X, Y=None):
            self.X = X
            self.Y = Y
            
        def __repr__(self):
            return f"{self.X} on {self.Y}"
        
        def __eq__(self, other):
            return (self.X, self.Y) == (other.X, other.Y)
        
    def __init__(self, blocks_world, blocks_world_goal, max_steps):
        self.blocks_world = blocks_world
        self.create_g(blocks_world_goal)
        self.last_state = copy.copy(self.blocks_world)
        self.num_steps = 0
        self.max_steps = max_steps
        
        self.B = self.blocks_world.num_blocks
        
        # The odds we took an action the previous timestep (which is 0 for everything)
        self.a_stack = np.zeros((self.B, self.B), dtype=float)
        self.a_unstack = np.zeros((self.B, self.B), dtype=float)
        
        # Create a grid of initial estimates for all c_0(X) and o_0(X, Y)
        self.previous_c_t_est = np.zeros(self.B, dtype=float)
        self.previous_o_t_est = np.zeros((self.B, self.B), dtype=float)
        
        # A block is on top of another block if it is in the same column with a higher row number
        # (so it also includes those not directly above)
        for col in range(0, self.B):
            for row in range(0, self.B):
                current_block = self.blocks_world.world[row, col]
                if current_block is not None:
                    self.previous_c_t_est[current_block.index] = 1 # Temporarily assume it's clear
                    if row < self.B: # Check all the blocks above
                        row = row + 1
                        next_block = self.blocks_world.world[row, col]
                        if next_block is not None:
                            self.previous_o_t_est[next_block.index, current_block.index] = 1
                            self.previous_c_t_est[current_block.index] = 0 # Was not clear, so put back to 0
                        #else: 
                        #    break # If the block is empty, the rest of the column is empty
                else:
                     break # Stop checking this column if we hit an empty space
                     
        self.current_c_t_est = copy.copy(self.previous_c_t_est)
        self.current_o_t_est = copy.copy(self.previous_o_t_est)
        
    def create_g(self, blocks_world_goal):
        self.g = []
        for b in range(0, blocks_world_goal.num_blocks):
            if blocks_world_goal.blocks[b].block_below is not None:
                self.g.append(NoPlanSolver.Goal(b, blocks_world_goal.blocks[b].block_below.index))            
            
    def get_c_t(self, X):
        # Odds X is clear of blocks on top
        # Probability given by paper:
        ### c_t(X) = 1 - (1 / ||B||) * sum(o_t(Y, X) for all Y in B)
        # Fixed probability:
        # c_t(X) = 1 - sum(o_t(Y, X) for all Y in B)
        sum = 0.0
        for Y in range(0, self.B):
            sum = sum + self.get_o_t(Y, X)
        #sum = 1 - (sum / self.B)
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
        self.current_c_t_est = np.zeros(self.B, dtype=float)
        self.current_o_t_est = np.zeros((self.B, self.B), dtype=float)
        
        for x in range(0, self.B):
            self.current_c_t_est[x] = self.get_c_t(x)
            
        for x in range(0, self.B):
            for y in range(0, self.B):
                self.current_o_t_est[x, y] = self.get_o_t(x, y)
            
    def get_gradient_a_stack(self, X, Y):
        # Somehow X on X is valid?
        return (1 if any(NoPlanSolver.Goal(X, Y) == goal for goal in self.g) else 0) * self.get_c_t_1(X) * self.get_c_t_1(Y)
        
    def get_gradient_a_unstack(self, X, Y):
        dgdo = 1 if any(NoPlanSolver.Goal(X, Y) == goal for goal in self.g) else 0
        dodc = (self.get_c_t_1(Y) * self.a_stack[X, Y]) - self.a_unstack[X, Y]
        dcdo = 0
        doda = 0
        for Z in range(0, self.B):
            # If Z is on X
            dcdo = self.get_o_t_1(Z, X)
            doda = self.get_c_t_1(Z)
            break;
        if dgdo == 1:
            print(f"Unstack {X} off {Y}: {dgdo} {dodc} {dcdo} {doda}")
        return dgdo * dodc * dcdo * doda
        
    def get_gradients(self):
        self.G = np.zeros(((2 * self.B), self.B), dtype=float)
        # The first ||B|| rows represent the gradients of a_stack
        for X in range(0, self.B):
            for Y in range(0, self.B):
                self.G[X, Y] = self.get_gradient_a_stack(X, Y)
        # The next ||B|| rows represent the gradients of a_unstack
        for X in range(0, self.B):
            for Y in range(0, self.B):
                self.G[X + self.B, Y] = self.get_gradient_a_unstack(X, Y)
    
    def next_time_step(self):
        self.create_estimates()
        self.previous_c_t_est = self.current_c_t_est
        self.previous_o_t_est = self.current_o_t_est
        
    def stack(self, src_block_idx, dst_block_idx):
        src_block = self.blocks_world.blocks[src_block_idx]
        dst_block = self.blocks_world.blocks[dst_block_idx]
        if self.blocks_world.can_stack(src_block, dst_block):
            self.a_stack = np.zeros((self.B, self.B), dtype=float)
            self.a_unstack = np.zeros((self.B, self.B), dtype=float)
            self.a_stack[src_block_idx, dst_block_idx] = 1
            self.blocks_world.stack(src_block, dst_block)
            self.next_time_step()
            
    def unstack(self, src_block_idx):
        src_block = self.blocks_world.blocks[src_block_idx]
        # There is no merit to unstacking if there's nothing below the block, so make sure there is something
        if self.blocks_world.can_unstack(src_block) and src_block.block_below is not None:
            self.a_stack = np.zeros((self.B, self.B), dtype=float)
            self.a_unstack = np.zeros((self.B, self.B), dtype=float)
            self.a_unstack[src_block_idx, src_block.block_below.index] = 1
            self.blocks_world.unstack(src_block)
            self.next_time_step()
        
    def perform_next_action(self):
        solved = True
        for goal in self.g:
            if self.blocks_world.blocks[goal.X].block_below is None:
                solved = False
                break
            if self.blocks_world.blocks[goal.X].block_below.index != goal.Y:
                print(f"Still need to stack {goal.X} on {goal.Y}")
                solved = False
                break
            
        if solved:
            print("Solved!")
            return False
        
        if self.num_steps >= self.max_steps:
            print("Reached max steps")
            return False
        
        max_gradient = 0
        argmaxesX = []
        argmaxesY = []
        self.get_gradients() # Updates the gradient estimates G
        for X in range(0, (2 * self.B)):
            for Y in range(0, self.B):
                if self.G[X, Y] > max_gradient:
                    argmaxesX = [X]
                    argmaxesY = [Y]
                    max_gradient = self.G[X, Y]
                elif self.G[X, Y] == max_gradient:
                    argmaxesX.append(X)
                    argmaxesY.append(Y)
                    
        # Select randomly from the set of maximal gradients
        if argmaxesX and argmaxesY and max_gradient > 0:
            X = rand.choice(argmaxesX)
            Y = rand.choice(argmaxesY)
            if X < self.B:
                print(f"Stacking {X} on {Y}")
                self.stack(X, Y)
            else:
                print(f"Unstacking {X}")
                self.unstack(X - self.B)
            self.num_steps = self.num_steps + 1
            self.blocks_world.print()
            return True
        else:
            print("Halted")
            return False
            
    def solve(self):
        while (self.perform_next_action()):
            print("Step taken")

def create_blocks_world(num_blocks, num_shuffle_steps, max_steps):
    global goal_world
    global shuffled_world
    global no_plan_solver
    
    blocks_world = BlocksWorld.BlocksWorld(num_blocks)
    goal_world = blocks_world
    print("Goal world:")
    blocks_world.print()
    
    print()
    
    print("Shuffled world:")
    shuffled_world = BlocksWorld.shuffle_world(blocks_world, num_shuffle_steps)
    shuffled_world.print()
    
    no_plan_solver = NoPlanSolver(shuffled_world, goal_world, max_steps)
    
if __name__ == "__main__":
    create_blocks_world(10, 20, 30)
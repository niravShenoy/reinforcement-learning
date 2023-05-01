import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn.functional as F

HEIGHT = 4
WIDTH = 4
TURNLEFT = {"north": "west", "east": "north", "south": "east", "west": "south"}
TURNRIGHT = {"north": "east", "east": "south", "south": "west", "west": "north"}

class Env:
    def __init__(self, pre_x, pre_y, pre_dir, post_x, post_y, post_dir, walls, pregrid_markers, postgrid_markers, height=HEIGHT, width=WIDTH):
        self.action_space = np.array(["putMarker", "turnLeft", "turnRight", "pickMarker", "move", "finish"])
        self.pre_x = pre_x
        self.pre_y = pre_y
        self.pre_dir = pre_dir
        self.walls = walls
        self.pregrid_markers = pregrid_markers.copy()
        self.state = State(pre_x, pre_y, pre_dir, walls, pregrid_markers.copy())
        self.goal_state = State(post_x, post_y, post_dir, walls, postgrid_markers.copy())
        self.height = height
        self.width = width
        self.observation_space = self.encodeState(self.state)
        self.reset()

    def reset(self):
        self.state = State(self.pre_x, self.pre_y, self.pre_dir, self.walls, self.pregrid_markers.copy())
        self.observation_space = self.encodeState(self.state)        
        return self.observation_space
        # raise NotImplementedError
    
    def step(self, action_index):
        action = self.action_space[action_index]
        reward = 0.0
        terminated = False
        if action == "move":
            reward = self.state.move()
        elif action == "turnLeft":
            reward = self.state.turnLeft()
        elif action == "turnRight":
            reward = self.state.turnRight()
        elif action == "pickMarker":
            reward = self.state.pickMarker()
        elif action == "putMarker":
            reward = self.state.putMarker()
        elif action == "finish":
            reward, terminated = self.state.finish(self.goal_state)
            truncated = not terminated
            if not truncated:
                self.observation_space = self.encodeState(self.state)
            return self.observation_space, reward, truncated, terminated
        else:
            raise ValueError("Invalid action")
        truncated = self.state.isTruncated()
        if not truncated:
            self.observation_space = self.encodeState(self.state)
        return self.observation_space, reward, truncated, terminated
        # raise NotImplementedError

    def render(self, goal_state = False):
        if goal_state:
            f = open("data_easy.txt", "a")
            f.write(str(self.state) + "\n")
            for i in range(self.height):
                for j in range(self.width):
                    if i == self.goal_state.x and j == self.goal_state.y:
                        f.write(self.goal_state.dir[0])
                    elif [i,j] in self.goal_state.walls:
                        f.write("#")
                    elif [i,j] in self.goal_state.markers:
                        f.write("o")
                    else:
                        f.write(".")
                f.write("\n\n")
            f.close()
        else:
            f = open("data_easy.txt", "a")
            f.write(str(self.state) + "\n")
            for i in range(self.height):
                for j in range(self.width):
                    if i == self.state.x and j == self.state.y:
                        f.write(self.state.dir[0])
                    elif [i,j] in self.state.walls:
                        f.write("#")
                    elif [i,j] in self.state.markers:
                        f.write("o")
                    else:
                        f.write(".")
                f.write("\n\n")
            f.close()
        # raise NotImplementedError

    def encodeState(self, state):
        encodedState = np.zeros((2, 2, self.height, self.width))
        for wall in state.walls:
            encodedState[0, 1, wall[0], wall[1]] = 1
        for marker in state.markers:
            encodedState[0, 1, marker[0], marker[1]] = 2
        if state.dir == "north":
            encodedState[0, 0, state.x, state.y] = 1
        elif state.dir == "west":
            encodedState[0, 0, state.x, state.y] = 2
        elif state.dir == "south":
            encodedState[0, 0, state.x, state.y] = 3
        elif state.dir == "east":
            encodedState[0, 0, state.x, state.y] = 4
        for wall in self.goal_state.walls:
            encodedState[1, 1, wall[0], wall[1]] = 1
        for marker in self.goal_state.markers:
            encodedState[1, 1, marker[0], marker[1]] = 2
        if self.goal_state.dir == "north":
            encodedState[1, 0, state.x, state.y] = 1
        elif self.goal_state.dir == "west":
            encodedState[1, 0, state.x, state.y] = 2        
        elif self.goal_state.dir == "south":
            encodedState[1, 0, state.x, state.y] = 3
        elif self.goal_state.dir == "east":
            encodedState[1, 0, state.x, state.y] = 4
        return F.one_hot(torch.tensor(encodedState).to(torch.int64), 5).cpu().numpy()

class State:

    TRUNCATE = 0
    TERMINATE = 5.0
    MOVE = 0
    TURN = 0
    PICKUP = 0.0
    INCORRECT_PICKUP = -1.0
    PUTDOWN = 0.0
    INCORRECT_PUTDOWN = -1.0

    def __init__(self, x, y, direction, walls, markers):
        self.x = x
        self.y = y
        self.dir = direction
        self.walls = walls
        self.markers = markers

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.dir == other.dir and self.markers == other.markers

    def __hash__(self):
        return hash((self.x, self.y, self.dir, tuple(self.markers)))

    def __str__(self):
        return "x: {}, y: {}, dir: {}, markers: {}".format(self.x, self.y, self.dir, self.markers)

    def __repr__(self):
        return self.__str__()

    def isTerminated(self, goal_state):
        return self == goal_state

    def isTruncated(self):
        if self.x < 0 or self.x >= HEIGHT or self.y < 0 or self.y >= WIDTH:
            return True
        if [self.x, self.y] in self.walls:
            return True
        return False

    def move(self):
        if self.dir == "north":
            self.x -= 1
            if self.x < 0 or [self.x, self.y] in self.walls:
                return self.TRUNCATE
        elif self.dir == "west":
            self.y -= 1
            if self.y < 0 or [self.x, self.y] in self.walls:
                return self.TRUNCATE
        elif self.dir == "south":
            self.x += 1
            if self.x >= HEIGHT or [self.x, self.y] in self.walls:
                return self.TRUNCATE
        elif self.dir == "east":
            self.y += 1
            if self.y >= WIDTH or [self.x, self.y] in self.walls:
                return self.TRUNCATE
        return self.MOVE

    def turnLeft(self):
        self.dir = TURNLEFT[self.dir]
        return self.TURN

    def turnRight(self):
        self.dir = TURNRIGHT[self.dir]
        return self.TURN

    def pickMarker(self):
        if [self.x, self.y] in self.markers:
            self.markers.remove([self.x, self.y])
            return self.PICKUP
        else:
            return self.INCORRECT_PICKUP

    def putMarker(self):
        if [self.x, self.y] not in self.markers:
            self.markers.append([self.x, self.y])
            return self.PUTDOWN
        else:
            return self.INCORRECT_PUTDOWN

    def finish(self, goal_state):
        if self.isTerminated(goal_state):
            return self.TERMINATE, True
        else:
            return self.TRUNCATE, False
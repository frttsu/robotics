#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('nbagg')
import sys
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np
from enum import Enum


# In[2]:


class Mode(Enum):
    STATE_TRANSITION = 1
    STRAIGHT_TRANSITION = 2
    SHIFT_TRANSITION = 3

class World:
    def __init__(self, time_span, time_interval, debug=True):
        self.objects = []  
        self.debug = debug
        self.time_span = time_span  
        self.time_interval = time_interval 

        
    def append(self,obj):  
        self.objects.append(obj)
    
    def draw(self): 
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')             
        ax.set_xlim(-250,250)                  
        ax.set_ylim(-250,250)
        #ax.set_xlim(-5,5)
        #ax.set_ylim(-5,5)
        ax.set_xlabel("X",fontsize=10)                 
        ax.set_ylabel("Y",fontsize=10)                 
        elems = []
        
        if self.debug:        
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                     frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()
        
    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*i)
        elems.append(ax.text(-245.4, 220, time_str, fontsize=10))
        #elems.append(ax.text(-4.4, 4.5, time_str,fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)    

class rob:
    def __init__(self,pose,agent):
        self.pose = pose
        self.agent = agent
        
        self.obs_stuck = 0
        self.obs_sign = 0
        
        self.angle = 3
        
        self.a = 0
        self.b = 0
    
    def draw(self,ax, elems):
        pass
    
    def state_transition(self,nu,omega,time,pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0), 
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )
    #agentXクラスのもとでつかうことを想定          
    def transition(self, nu,omega, time,obs):
        if self.agent.mode == Mode.STATE_TRANSITION:
            return self.state_transition(nu,omega,time,self.pose)
        elif self.agent.mode == Mode.STRAIGHT_TRANSITION:
            return self.straight_transition(nu,omega, time, obs)
        elif self.agent.mode == Mode.SHIFT_TRANSITION:
            return self.shift_transition(nu,omega,time, obs)
        else:
            return self.state_transition(nu,omega,time, self.pose)
        
    def sensor_return(self,obs):
        if obs:         
            if np.abs(obs[0][0][1]) > self.angle /180 * math.pi :
                self.obs_stuck = np.abs(obs[0][0][1]) - self.angle / 180 * math.pi
                self.obs_sign = np.sign(obs[0][0][1])
                return  self.obs_sign * self.angle / 180 * math.pi
            else:
                return obs[0][0][1]

        if self.obs_stuck > self.angle / 180 * math.pi:
            self.obs_stuck = self.obs_stuck - self.angle / 180 * math.pi
            return self.obs_sign *  self.angle / 180 * math.pi
        else:
            self.a = self.obs_stuck
            self.b = self.obs_sign
            self.obs_stuck = 0
            self.obs_sign = 0
            return self.a *self.b / 180 * math.pi
   #agentXクラスのもとでつかうことを想定                         
    def straight_transition(self,nu,omega,time,obs):
        if obs:  
            if obs[0][0][0] < self.agent.distance_minimum:
                self.agent.decelerate_nu()
                self.agent.keep_straight_change()
                

        t0 = self.pose[2]        
            
        if math.fabs(omega) < 1e-10:
            return self.pose + np.array( [nu* math.cos(t0)*time, 
                             nu *math.sin(t0)*time,
                             omega*time  + self.sensor_return(obs)] )
        else:
            return self.pose + np.array( [nu  / omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                             nu  / omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                             omega*time + self.sensor_return(obs) ] )
        
    #agentXクラスのもとでつかうことを想定                      
    def shift_transition(self, nu, omega, time, obs):
        if obs:
            if obs[0][0][0] < self.agent.distance_minimum:
                self.agent.decelerate_nu
                self.agent.keep_shift_change()
            
        t0 = self.pose[2]
        
        if math.fabs(omega) < 1e-10:
            return self.pose + np.array( [nu *math.cos(t0)*time+2.0, 
                                 nu *math.sin(t0)*time,
                                 omega*time+self.sensor_return(obs) ] )/ math.sqrt((nu * self.accelerate_rate *math.cos(t0)*time+2.0)**2 +(nu * self.accelerate_rate *math.sin(t0)*time)**2) * math.sqrt((nu * self.accelerate_rate *math.cos(t0)*time)**2 +(nu * self.accelerate_rate *math.sin(t0)*time)**2) 
        else:
            return self.pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0))+2.0, 
                                 nu  /omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                 omega*time+self.sensor_return(obs) ] )/ math.sqrt((nu/omega*(math.sin(t0 + omega*time) - math.sin(t0))+2.0)**2 + (nu* self.accelerate_rate/omega*(-math.cos(t0 + omega*time) + math.cos(t0))**2 )) * math.sqrt((nu* self.accelerate_rate/omega*(math.sin(t0 + omega*time) - math.sin(t0)))**2 + (nu* self.accelerate_rate/omega*(-math.cos(t0 + omega*time) + math.cos(t0))**2 ))

# In[3]:

class IdealRobot(rob):   
    def __init__(self, pose, agent=None, sensor=None, color="black"):    # 引数を追加
        super().__init__(pose,agent)
        self.pose = pose
        self.r = 10  
        self.color = color 
        self.agent = agent
        self.poses = [pose]
        self.id = 0
        self.sensor = sensor    # 追加
    
    def draw(self, ax, elems):         ### call_agent_draw
        x, y, theta = self.pose  
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x,xn], [y,yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        elems.append(ax.add_patch(c))
        self.poses.append(self.pose)
        elems.append(ax.text(self.pose[0]-20, self.pose[1]-20, "child AUV" + str(self.id), fontsize=8))         
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):                               #以下2行追加   
            self.agent.draw(ax, elems)
         
#     @classmethod           
#     def state_transition(cls, nu, omega, time, pose):
#         t0 = pose[2]
#         if math.fabs(omega) < 1e-10:
#             return pose + np.array( [nu*math.cos(t0), 
#                                      nu*math.sin(t0),
#                                      omega ] ) * time
#         else:
#             return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
#                                      nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
#                                      omega*time ] )

    def one_step(self, time_interval):
        if not self.agent: return        
        obs =self.sensor.data(self.pose) if self.sensor else None #追加
        nu, omega = self.agent.decision(obs) #引数追加
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        if self.sensor: self.sensor.data(self.pose)   
            
# In[4]:


class Agent: 
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega
        self.mode = Mode.STATE_TRANSITION
        
    def decision(self, observation=None):
        return self.nu, self.omega

class AgentX:
    def __init__(self, nu, omega,accelerate_rate, distance_minimum,distance_maximum):
        self.nu = nu
        self.omega = omega
        
        self.mode = Mode.STATE_TRANSITION
        
        self.accelerate_rate = accelerate_rate
        self.shift_switch = False
        self.keep_straight = False
        self.keep_shift = False
        
        self.distance_maximum = distance_maximum
        self.distance_minimum = distance_minimum
        if(distance_minimum > distance_maximum):
            print("input error")
            sys.exit()
            
        
    def decision(self, observation=None):
        if observation:
            self.mode = self.mode_change(observation)
        return self.nu, self.omega
    
    def shift_switch_change(self):
        self.shift_switch = not self.shift_switch
    
    def keep_straight_change(self):
        self.keep_straight = not self.keep_straight
    
    def keep_shift_change(self):
        self.keep_shift = not self.keep_shift
        
    def accelerate_nu(self):
        self.nu = self.nu * self.accelerate_rate
    
    def decelerate_nu(self):
        self.nu = self.nu / self.accelerate_rate
        
    def mode_change(self,obs):
        if ((obs[0][0][0] > self.distance_maximum) and(self.shift_switch == False)) or (self.keep_straight == True) :
            if (self.keep_straight == False) :
                self.accelerate_nu()
                self.keep_straight_change()
            return Mode.STRAIGHT_TRANSITION
        elif ((obs[0][0][0] > self.distance_maximum) and (self.shift_switch == True)) or (self.keep_shift == True):
            if (self.keep_shift == False):
                self.accelerate_nu()
                self.nu = self.nu * self.accelerate_rate
            return Mode.SHIFT_TRANSITION
        else:
            return Mode.STATE_TRANSITION

        
class AgentY:
    def __init__(self,time_interval, data):
        self.nu = 0
        self.omega = 0
        self.time_interval = time_interval
        self.data = data
        self.iterator = iter(self.data)
        self.current_data = []
        self.time = 0
        self.mode = Mode.STATE_TRANSITION
        
    def decision(self, observation=None):
        self.data_change()
        return self.nu, self.omega
        
    def data_change(self):
        if(self.time <= 1e-10):
            self.current_data = next(self.iterator, "end")
            if(self.current_data == "end"):
#                 print("byebye")
                return
            print(self.current_data)
            self.nu = self.current_data[0]
            self.omega = self.current_data[1]
            self.time = self.current_data[2]
        self.time = self.time - self.time_interval
#        print(self.time)
            
# In[5]:


class Landmark:
    def __init__(self, pos , agent = None):
        self.pos = pos
        self.agent = agent
        self.id = None
        
    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=30, marker="*", label="landmarks", color="orange")
        elems.append(c)
#         elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))
        elems.append(ax.text(self.pos[0]-20, self.pos[1]-20, "parent AUV" + str(self.id), fontsize=8))
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0), 
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )

    def one_step(self, time_interval):
        if not self.agent: return        
        nu, omega = self.agent.decision() #引数追加
        self.pos = self.state_transition(nu, omega, time_interval, self.pos) 


# In[6]:


class Map:
    def __init__(self):       # 空のランドマークのリストを準備
        self.objects = []
        
    def append_object(self, ob):       # ランドマークを追加
        ob.id = len(self.objects)           # 追加するランドマークにIDを与える
        self.objects.append(ob)

    def draw(self, ax, elems):                 # 描画（Landmarkのdrawを順に呼び出し）
        for ob in self.objects: ob.draw(ax, elems)

    def one_step(self,time_interval):
        for ob in self.objects: ob.one_step(time_interval)
                         


# In[7]:


class IdealCamera:
    def __init__(self, env_map, distance_range=(0.5, 350), direction_range=(-math.pi, math.pi)):
        self.map = env_map
        self.lastdata = []
        
        self.distance_range = distance_range
        self.direction_range = direction_range
        
    def visible(self, polarpos):  # ランドマークが計測できる条件
        if polarpos is None:
            return False
        #print(polarpos[0])
        #print(self.distance_range[0])
        #print(self.distance_range[1])
        #print(self.distance_range[0] <= polarpos[0] and polarpos[0] <= self.distance_range[1])
    
        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1]                 and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]
        
    def data(self, cam_pose):
        observed = []
        for lm in self.map.objects:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):               # 条件を追加
                observed.append((z, lm.id))   # インデント
            
        self.lastdata = observed
        return observed
    
    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        #print("obj=",obj_pos)
        #print("cam=",cam_pose)
        diff = obj_pos[0:2]- cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array( [np.hypot(*diff), phi ] ).T
    
    def draw(self, ax, elems, cam_pose): 
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x,lx], [y,ly], color="pink")


# In[8]:


if __name__ == '__main__':   ###name_indent
    world = World(30, 0.1) 

    ### 地図を生成して3つランドマークを追加 ###
    m = Map()                                  
    m.append_landmark(Landmark(2,-2))
    m.append_landmark(Landmark(-1,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)          

    ### ロボットを作る ###
    straight = Agent(0.2, 0.0)    
    circling = Agent(0.2, 10.0/180*math.pi)  
    robot1 = IdealRobot( np.array([ 2, 3, math.pi/6]).T,    sensor=IdealCamera(m), agent=straight )             # 引数にcameraを追加、整理
    robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")  # robot3は消しました
    world.append(robot1)
    world.append(robot2)

    ### アニメーション実行 ###
    world.draw()
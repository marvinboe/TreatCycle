########################################
#  Copyright 2016 Marvin A. Böttcher
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#########################################

import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

print("module loaded")

class draw_egt_dynamics:
    'draws dynamics of given interaction matrix and fixed points into triangle'
    #corners of triangle and calculation of points
    r0=np.array([0,0])
    r1=np.array([1,0])
    r2=np.array([1/2.,np.sqrt(3)/2.])
    corners =np.array([r0,r1,r2])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=5)
    trimesh_fine = refiner.refine_triangulation(subdiv=5)

    def __init__(self,fun):
        self.f=fun
        self.calculate_stationary_points()
        self.calc_direction_and_strength()

    #barycentric coordinates
    def xy2ba(self,x,y):
        corner_x=self.corners.T[0]
        corner_y=self.corners.T[1]
        x_1=corner_x[0]
        x_2=corner_x[1]
        x_3=corner_x[2]
        y_1=corner_y[0]
        y_2=corner_y[1]
        y_3=corner_y[2]
        l1=((y_2-y_3)*(x-x_3)+(x_3-x_2)*(y-y_3))/((y_2-y_3)*(x_1-x_3)+(x_3-x_2)*(y_1-y_3))
        l2=((y_3-y_1)*(x-x_3)+(x_1-x_3)*(y-y_3))/((y_2-y_3)*(x_1-x_3)+(x_3-x_2)*(y_1-y_3))
        l3=1-l1-l2
        return np.array([l1,l2,l3])

    def ba2xy(self,x):
        ### x: array of 3-dim ba coordinates
        ### corners: coordinates of corners of ba coordinate system
        return self.corners.T.dot(x.T).T


    def calculate_stationary_points(self):
        fp_raw=[]
        border=270
        for x,y in zip(self.trimesh_fine.x[border:-border], self.trimesh_fine.y[border:-border]):
            start=self.xy2ba(x,y)
            delta=1e-12
            fp_try=np.array([])
            # if np.allclose(self.f(start,0),np.array([0,0,0]),atol=delta,rtol=delta):
            #     #print(start)
            #     fp_try=start
            # else:
            sol=scipy.optimize.root(self.f,start,args=(0,))#,xtol=1.49012e-10,maxfev=1000
            if sol.success:
                fp_try=sol.x
            else:
                continue
            #print (start,fp_try)
            if np.all((fp_try>-delta) & (fp_try <1+delta)):
                if not np.array([np.allclose(fp_try,x,atol=1e-5) for x in fp_raw]).any():
                    fp_raw.append(fp_try.tolist())
        fp_raw=np.array(fp_raw)
        # print(fp_raw)
        self.fixpoints=self.corners.T.dot(np.array(fp_raw).T).T

    def calc_direction_and_strength(self):
        direction= [self.f(self.xy2ba(x,y),0) for x,y in zip(self.trimesh.x, self.trimesh.y)]
        self.direction_norm=np.array([self.ba2xy(v)/np.linalg.norm(v) if np.linalg.norm(v)>0 else np.array([0,0]) for v in direction])
        self.direction_norm=self.direction_norm
        #print(direction_ba_norm)
        self.pvals =[np.linalg.norm(v) for v in direction]

    def plot_simplex(self,ax,cmap='viridis',**kwargs):

        ax.triplot(self.triangle,linewidth=0.8,color="black")
        ax.tricontourf(self.trimesh, self.pvals, alpha=0.8, cmap=cmap,**kwargs)
        # Q = ax.quiver(self.trimesh.x, self.trimesh.y, self.direction_norm.T[0],self.direction_norm.T[1],self.pvals,angles='xy',pivot='tail',  cmap=cmap)#pivot='mid',
        Q = ax.quiver(self.trimesh.x, self.trimesh.y, self.direction_norm.T[0],self.direction_norm.T[1],angles='xy',pivot='tail')#pivot='mid',
        ax.axis('equal')
        ax.axis('off')

        #timescatter=ax.scatter(points[::5,0],points[::5,1],c=t[::5],linewidth=0.0,cmap='viridis',alpha=.5)
        ax.scatter(self.fixpoints[:,0],self.fixpoints[:,1],c="black",s=70,linewidth=0.3)
        #fig.colorbar(timescatter,label="time")
        ax.annotate("D",(0,0),xytext=(-0.0,-0.15),horizontalalignment='center')
        ax.annotate("G",(1,0),xytext=(1.0,-0.15),horizontalalignment='center')
        ax.annotate("AG",self.corners[2],xytext=self.corners[2]+np.array([0.01,0.06]),horizontalalignment='center')


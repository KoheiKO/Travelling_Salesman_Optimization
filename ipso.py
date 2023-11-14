import random
import math
import matplotlib.pyplot as plt
from util import City, read_cities, write_cities_and_return_them, generate_cities, path_cost
import time
import numpy as np

class Particle:
    def __init__(self, route, cost=None):
        self.route = route
        self.pbest = route
        self.current_cost = cost if cost else self.path_cost()
        self.pbest_cost = cost if cost else self.path_cost()
        self.velocity = []

    def clear_velocity(self):
        self.velocity.clear()

    def update_costs_and_pbest(self):
        self.current_cost = self.path_cost()
        if self.current_cost < self.pbest_cost:
            self.pbest = self.route
            self.pbest_cost = self.current_cost

    def path_cost(self):
        return path_cost(self.route)


class PSO:

    def __init__(self, iterations, population_size, gbest_probability=1.0, pbest_probability=1.0, cities=None):
        self.cities = cities
        self.gbest = None
        self.gcost_iter = []
        self.iterations = iterations
        self.population_size = population_size
        self.particles = []
        self.gbest_probability = gbest_probability
        self.pbest_probability = pbest_probability

        solutions = self.initial_population()
        self.particles = [Particle(route=solution) for solution in solutions]

    def random_route(self):
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        random_population = [self.random_route() for _ in range(self.population_size - 1)]
        greedy_population = [self.greedy_route(0)]
        return [*random_population, *greedy_population]
        # return [*random_population]

    def greedy_route(self, start_index):
        unvisited = self.cities[:]
        del unvisited[start_index]
        route = [self.cities[start_index]]
        while len(unvisited):
            index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
            route.append(nearest_city)
            del unvisited[index]
        return route
    
    def route_order(self, route):
        
        coor_route = []
        # print( len(city))
        for i  in range(len(self.cities)):
            for j in range(len(route)):
                if route[j] == self.cities[i]:
                    coor_route.append(j) 
        return(coor_route)
    
    def route_coordinate(self,order):
        route_coordinate = []
        cities = self.cities
        # print("cities = " ,cities) 
        # print("len(order) = ", len(order))
        for i in range(len(order)):
            
            coordinate = cities[order[i]]
            route_coordinate.append(coordinate)

        return route_coordinate


    def culc_dist(self, route):
        dist_sum = 0
        cities = self.cities
        j = 0
        # print(type(cities))
        for i in range(len(route)-1):
            city1 = cities[route[i]]
            city2 = cities[route[i+1]]
            x1, y1 = city1[0] , city1[1]
            x2, y2 = city2[0] , city2[1]  
            
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            dist_sum += distance
            j += 1
            # print("len(route) - 2 = " , len(route) - 2)
            # print("i = " , i)

            # print("route = " ,route)
            
            # print("city1 = " ,city1)
            # print(type(city1))
            # print("city1[0] = " ,city1[0])
        city_start = cities[route[0]]
        city_fin = cities[route[j]]
        x1, y1 = city_start[0] , city_start[1]
        x2, y2 = city_fin[0] , city_fin[1]  
        
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        dist_sum  += distance

       
        return dist_sum
    


    def insert(self,origin,insert_route):
        route_dist = []
        route_ap = []
        route = []
        # route_min = []
        ## insert pb_dd into x_d
        # print("len(origin) = " , len(origin))
        for i in range(len(origin)):
            if i != len(origin) :
                route = origin [0:i+1] + insert_route + origin [i+1: len(origin) ]
            else:
                route = origin [0:len(origin)] + insert_route

            # print("route = " , route)
            route_ap.append(route)
            route_dist.append(pso.culc_dist(route)) 
            
        route_dist_min = min(route_dist) 
        min_index =   route_dist.index(route_dist_min)
        route_min = route_ap[min_index]
       
        
        return route_min,route_dist_min


    
    def run(self):
        # self.gbest = min(self.particles, key=lambda p: p.pbest_cost)

        # ランダムに並び替えたリストを格納するリスト
        all_particle_best = []
        dist_list = []

        update_freq =  10
        fig = plt.figure(1)
        ax0 = fig.add_subplot(1, 2, 1) 
        ax1 = fig.add_subplot(1, 2, 2)
        
        
        ax0.clear() 

        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-25, 25)
        ax1.clear()
            # fig = plt.figure(1)
        

        # 100行のリストを用意
        for _ in range(pso.population_size):
            # 1から50までの数字のリストを生成
            numbers = list(range(0, len(self.cities)))
            
            # リストをランダムに並び替え
            random.shuffle(numbers)
            
            # シャッフルされたリストをrandom_listsに追加
            all_particle_best.append(numbers)

            dist_list.append(pso.culc_dist(numbers))
        # print(all_particle_best[99])
        # print(dist_list)

        gbest_dist = min(dist_list)
        gbest_route_index = dist_list.index(gbest_dist)
        gbest_route = all_particle_best[gbest_route_index]
        # print("gbest_dist = ", gbest_dist)
        # print("gbest_route_index = ", gbest_route_index)
        # print("gbest_route = ", gbest_route)
        # x_list =  pso.route_coordinate(gbest_route)
        # print("x_list = ", x_list)
        # print("x_list = ", x_list[1])
        

        ite = 1
        # print("all_particle_best = ", all_particle_best)
        
        c1 = 0.7 
        c2 = 0.1
        r1 , r2 = np.random.rand() , np.random.rand() 
        n = len(self.cities)

        print(r1 , r2)
        
        # print("self.particles = ", self.particles)

        print(f"initial cost is {gbest_dist}")
        # print(f"initial cost is {self.gbest.pbest_cost}")
        gbest_dist_plot, ite_plot = [], []
        for t in range(self.iterations):
            print("iteration = ", ite)
            
            particle_num = 0
            # time.sleep(1) 

            gbest_dist = min(dist_list)
            gbest_route_index = dist_list.index(gbest_dist)
            gbest_route = all_particle_best[gbest_route_index]
            gbest_coordinate = pso.route_coordinate(gbest_route)


            

            if (t+1) % update_freq == 0:
                # plt.figure(0)
                gbest_dist_plot.append(gbest_dist)
                ite_plot.append(ite)
                # ax1.set_title(f'pso TSP iter {t}')
                ax1.set_title('IPSO iter')
                # ax0.set_xlim(0, 1000)
                # ax0.set_ylim(0, self.iterations)
                ax0.plot(ite_plot, gbest_dist_plot, 'g')
                ax0.set_ylabel('Distance')
                ax0.set_xlabel('Generation')



                x_list, y_list = [], []
                for i in range(len(gbest_route)):
                    city = gbest_coordinate[i]
                    x_list.append(city[0])
                    y_list.append(city[1])
                x_list.append(x_list[0])
                y_list.append(y_list[0])

                
                if t > self.iterations -update_freq -1:
                  
                    ax1.cla()  # Clear the figure
                    ax1.plot(x_list, y_list, 'ro')
                    ax1.plot(x_list, y_list, 'b')

                    # ax1.draw()
                    # ax1.pause(0.001)
                    # ax1.show(block=False)  
                    # time.sleep(100)
                else:
                   
                    ax1.cla()  # Clear the figure
                    ax1.plot(x_list, y_list, 'ro')
                    ax1.plot(x_list, y_list, 'g')
                    # ax1.draw()

                ax1.set_title(f'pso TSP iter {t+1}')
                ax1.grid()
                
                plt.pause(0.001)
                plt.show(block=False)  # Set block to False to allow code execution to continue

        

            for particle in self.particles:
                
                
                new_route = all_particle_best[particle_num]


                r1 = int(np.round(c1 * np.random.rand() * (n + 1)))
                while r1 < 4:
                    r1 = int(np.round(c1 * np.random.rand() * (n + 1)))
                    

                r2 = int(np.round(c2 * np.random.rand() * (n + 1)))
                while r2 < 2:
                    r2 = int(np.round(c2 * np.random.rand() * (n + 1)))
                    
                sr1 = np.random.randint(n)
                sr2 = np.random.randint(n)
                # print("sr1 = " ,sr1)
                # print("r1 = " , r1)

                # if i > 0 :
                # print("particle.pbest = " ,particle.pbest)
                pbest_route = all_particle_best[particle_num]
                # print("pbest_route = " , pbest_route)
               
                # print("pbest_route = " , pbest_route)
                # print("gbest_route = " , gbest_route)
                pb = pbest_route
                pb_d = pbest_route[sr1:sr1 + r1] if sr1 + r1 - 1 <= n else pbest_route[sr1:n] + pbest_route[0:sr1 + r1 - n]
                
                # gbest_route   = self.gbest.pbest[:]
                
                lb = gbest_route
                lb_d = gbest_route[sr2:sr2 + r2] if sr2 + r2 - 1 <= n else gbest_route[sr2:n] + gbest_route[0:sr2 + r2 - n]
                pb_dd = [x for x in pb_d if x not in lb_d]

                 
                
                
                # print("pb = " , pb_d )
                # print("pb_dd = " , pb_dd )
                # print("ld = ", lb_d)
                    
                x = new_route
                # print("x = " , x)
                x_subs_lb_d = [i for i in x if i not in lb_d]
                x_d = [i for i in x_subs_lb_d if i not in pb_dd]
                # print("x_d = " , x_d)

                # print("x_d = ", x_d)
                
                x_dd , x_dd_dist = pso.insert(x_d,pb_dd)

                
                # print("x = " , x)
                # print("pb_d  = " , pb_d)
                # print("lb_d  = " , lb_d)
                # print("pb_dd  = " , pb_dd)
                # print("x_d  = " , x_d)
                # print("x_dd  = " , x_dd)


               
                x_new , x_new_dist = pso.insert(x_dd,lb_d)
                # print("x_new = " , x_new)
                # print("x_new[3:8] = " , x_new[3:8])


                all_particle_best [particle_num]= x_new
                # print("all_particle_best = " , all_particle_best)
                x_new_dist = pso.culc_dist(x_new)
                if x_new_dist < dist_list[particle_num]:
                    all_particle_best[particle_num] = x_new
                    dist_list[particle_num] = x_new_dist

                

                # particle.route = pso.route_coordinate(x_new)
                # particle.update_costs_and_pbest()
                # print(particle.path_cost())

                time.sleep(0) 
                particle_num += 1
                
            ite += 1
            time.sleep(0) 

            
if __name__ == "__main__":
    cities = read_cities()
    pso = PSO(iterations=1000, population_size=10, pbest_probability=0.9, gbest_probability=0.02, cities=cities)
    pso.run()

    time.sleep(1000)


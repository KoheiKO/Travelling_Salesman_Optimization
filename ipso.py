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
            for j in range(len(self.cities)):
                if route[j] == self.cities[i]:
                    coor_route.append(j) 
        return(coor_route)
    
    def route_coordinate(self,order):
        route_coordinate = []
        cities = self.cities
        # print("cities = " ,cities) 
        # print("len(order) = ", len(order))
        for i in range(len(order)-1):
            
            coordinate = cities[order[i]]
            route_coordinate.append(coordinate)

        return route_coordinate


    def calc_dist(self, route):
        dist_sum = 0
        cities = self.cities
        # print(type(cities))
        for i in range(len(route)-1):
            city1 = cities[route[i]]
            city2 = cities[route[i+1]]
            x1, y1 = city1[0] , city1[1]
            x2, y2 = city2[0] , city2[1]  
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            dist_sum += distance

            # print("len(route) - 2 = " , len(route) - 2)
            # print("i = " , i)

            # print("route = " ,route)
            
            # print("city1 = " ,city1)
            # print(type(city1))
            # print("city1[0] = " ,city1[0])
        return dist_sum
    


    def insert(self,origin,insert_route):
        route_dist_min = np.Inf
        ## insert pb_dd into x_d
        for i in range(len(origin) + len(insert_route)):
            if i != len(origin) + len(insert_route):
                route = origin [0:i] + insert_route + origin [i: len(origin) + len(insert_route) -1]
            else:
                route = origin [0:i] + insert_route
            
            route_dist = pso.calc_dist(route) 
            

            if route_dist < route_dist_min:
                route_dist_min = route_dist
                route_min = route

        
        return route_min,route_dist_min



    def run(self):
        

        # ランダムに並び替えたリストを格納するリスト
        all_particle_best = []

        # 100行のリストを用意
        for _ in range(100):
            # 1から50までの数字のリストを生成
            numbers = list(range(0, len(self.cities)))
            
            # リストをランダムに並び替え
            random.shuffle(numbers)
            
            # シャッフルされたリストをrandom_listsに追加
            all_particle_best.append(numbers)
        # print(all_particle_best[99])

        time.sleep(2) 
        ite = 1
        
        
        c1 = 0.7 
        c2 = 0.1
        r1 , r2 = np.random.rand() , np.random.rand() 
        n = len(self.cities)

        print(r1 , r2)
        self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
        print(f"initial cost is {self.gbest.pbest_cost}")
        plt.ion()
        plt.draw()
        for t in range(self.iterations):
            print("iteration = ", ite)
            particle_num = 1
            
            # time.sleep(1) 


            self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
            # print("gbest = ", self.cities)
            if t % 20 == 0:
                plt.figure(0)
                plt.plot(pso.gcost_iter, 'g')
                plt.ylabel('Distance')
                plt.xlabel('Generation')
                fig = plt.figure(0)
                fig.suptitle('pso iter')
                x_list, y_list = [], []
                for city in self.gbest.pbest:
                    x_list.append(city.x)
                    y_list.append(city.y)
                x_list.append(pso.gbest.pbest[0].x)
                y_list.append(pso.gbest.pbest[0].y)
                fig = plt.figure(1)
                fig.clear()
                fig.suptitle(f'pso TSP iter {t}')

                plt.plot(x_list, y_list, 'ro')
                plt.plot(x_list, y_list, 'g')
                plt.draw()
                plt.pause(.001)
            self.gcost_iter.append(self.gbest.pbest_cost)

            for particle in self.particles:
                print("particle_num = ", particle_num)
                
                # print("Iteration" , particle)
                # particle.clear_velocity()
                # temp_velocity = []
                gbest = self.gbest.pbest[:]
                new_route = particle.route[:]
                
                print("new_route = ", new_route)
                # time.sleep(1) 

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
                pbest_route = pso.route_order(particle.pbest)
                print("pbest_route = " , pbest_route)
                gbest_route = pso.route_order(gbest)
                # print("pbest_route = " , pbest_route)
                # print("gbest_route = " , gbest_route)
                pb = pbest_route
                pb_d = pbest_route[sr1:sr1 + r1] if sr1 + r1 - 1 <= n else pbest_route[sr1:n] + pbest_route[:sr1 + r1 - 1 - n]
                

                
                lb = gbest_route
                lb_d = gbest_route[sr2:sr2 + r2] if sr2 + r2 - 1 <= n else gbest_route[sr2:n] + gbest_route[:sr2 + r2 - 1 - n]
                pb_dd = [x for x in pb_d if x not in lb_d]

                 
                
                
                # print("pb = " , pb_d )
                # print("pb_dd = " , pb_dd )
                # print("ld = ", lb_d)
                    
                x = pso.route_order(new_route)
                x_subs_lb_d = [i for i in x if i not in lb_d]
                x_d = [i for i in x_subs_lb_d if i not in pb_dd]
               

                # print("x_d = ", x_d)
                
                x_dd , x_dd_dist = pso.insert(x_d,pb_dd)

                

                print("x_d  = " , x_d)
                print("x_dd  = " , x_dd)
                print("lb_d  = " , lb_d)
                print("pb_dd  = " , pb_dd)
               
                x_best , x_best_dist = pso.insert(x_dd,lb_d)
                print("x_best = " , x_best)




                # print("x_dd_min_dist = ", x_dd_dist)
                
                
                # for i in range(n):

                #     # print("particle.pbest = " , particle.pbest)

                    

                #     if new_route[i] != particle.pbest[i]:
                #         swap = (i, particle.pbest.index(new_route[i]), self.pbest_probability)
                #         temp_velocity.append(swap)
                #         new_route[swap[0]], new_route[swap[1]] = \
                #             new_route[swap[1]], new_route[swap[0]]

                # for i in range(n):
                #     if new_route[i] != gbest[i]:
                #         swap = (i, gbest.index(new_route[i]), self.gbest_probability)
                #         temp_velocity.append(swap)
                #         gbest[swap[0]], gbest[swap[1]] = gbest[swap[1]], gbest[swap[0]]

                # particle.velocity = temp_velocity

                # for swap in temp_velocity:
                #     if random.random() <= swap[2]:
                #         new_route[swap[0]], new_route[swap[1]] = \
                #             new_route[swap[1]], new_route[swap[0]]
                # print(pso.route_coordinate(x_best))

                particle.route = pso.route_coordinate(x_best)
                particle.update_costs_and_pbest()
                # print(particle.path_cost())

                
                particle_num += 1
            ite += 1

            
if __name__ == "__main__":
    cities = read_cities()
    pso = PSO(iterations=100, population_size=100, pbest_probability=0.9, gbest_probability=0.02, cities=cities)
    pso.run()
    
    # particle = Particle()
    print(f'cost: {pso.gbest.pbest_cost}\t| gbest: {pso.gbest.pbest}')

    x_list, y_list = [], []
    for city in pso.gbest.pbest:
        x_list.append(city.x)
        y_list.append(city.y)
    x_list.append(pso.gbest.pbest[0].x)
    y_list.append(pso.gbest.pbest[0].y)
    fig = plt.figure(1)
    fig.suptitle('pso TSP')

    plt.plot(x_list, y_list, 'ro')
    plt.plot(x_list, y_list)
    plt.show(block=True)


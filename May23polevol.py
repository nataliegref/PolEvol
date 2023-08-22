
import numpy as np
import random
import copy
from collections import Counter
import pickle
import csv 
from csv import writer
import os
import pandas as pd


def calc_fitness(new_species, form):
    """
    Takes in two variables: new_species (list), form (string or list).
    Uses the string 'form' to decide which fitness function to use.
    Calculates the fitness for each polymer within the list 'new_species', and appends them to a list 'fitness'.
    Returns one variables: fitness (list).
    """
    fitness = []
    for ns in new_species:
        c = Counter(ns.sequence)
        i = c['1']
#         print(f"seq: {ns.sequence}, i is {i}")
        if form[0] == 'linear':
            a = form[1]
            f = 1+(i/(len(ns.sequence)**a))
        elif form[0] == 'exp':
            b = form[1]
            f = 1+((math.exp(2*(b*i))-1)/(math.exp(2*(b*i))+1))
        elif form == 'log':
            f = math.log(i+1)
#             print(f"f is :{f}")
        elif form == 'og':
            f = 1
            t = 0
            for i in ns.sequence:
                t += int(i)
            if (len(ns.sequence)**0) == 0:
                print(f"len is {len(ns.sequence)} and a is {0}")
            else:
                t = (t**1)/(len(ns.sequence)**0)
            f += t
            
        elif form[0] == 'step5':
            a = form[1]
            if i < 5:
                f = 1
            elif i >= 5:
                f = 1+(i/(len(ns.sequence)**a))
    #exp
    #b = 0.02 or b = 0.1
        fitness.append(f)
    
    return fitness

def calc_probs(fitness):
    """
    Takes in one variable: fitness (list).
    For each entry in fitness, calculates the relative probability of that species dying (see def kill).
    Returns one variable: probs (list).
    """
    probs = []
        
    for species in fitness:
        probs.append(1/species)
        
    return probs

def populate_dict_list(key, value, d):
    if key in d.keys():
        d[key].append(value)
    else:
        d[key] = [value]
    #print(f"dict is {d}")
    return d

class Polymer:
    def __init__(self, seq, gen = 'X', num = 'X', parent = 'na', daughter = '', copy = 'na'):
        self.sequence = seq
        self.generation = gen
        self.ID = num
        if type(parent) != str:
            parent = Polymer(parent.sequence, parent.generation, parent.ID, 'old')
        self.parent = parent
        self.daughter = daughter
        self.copy = copy
        
    def __repr__(self):
        return f'{self.sequence}, born: {self.generation}, daughter: {self.daughter}'

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Polymer):
            return (self.sequence == other.sequence) and (self.generation == other.generation) and (self.ID == other.ID) and (self.parent == other.parent) and (self.daughter == other.daughter)
        return NotImplemented

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple(sorted(self.__dict__.items())))
            
    def replicate(self, new_seq):
        self.template = new_seq
        

class Run:
    def __init__(self,date,t,sel,spots,save_point,pool_size,rates,fitness,fitness_form,mutation,ratio,rate_name,m0,d_solution, d_surface, d_killed, d_events, time, max_gen, filename):
        self.date = date
        self.template = t
        self.selection = sel
        self.spots =  spots
        self.save_point =  save_point
        self.pool_size =  pool_size
        self.rates = rates
        self.fitness = fitness
        self.fitnessform = fitness_form 
        self.mutation = mutation 
        self.ratio = ratio 
        self.rate_name = rate_name
        self.m0 = m0
        self.solution = d_solution
        self.surface = d_surface
        self.killed = d_killed
        self.events = d_events
        self.time = time
        self.maxgen = max_gen
        self.name = filename
                
    def __repr__(self):
 
        return f'{str(self.date)}GSP_{self.template}_{self.selection}N{str(self.spots)}_add{str(self.maxgen)}_k{str(self.m0)}-{self.rate_name}_f{self.fitness}_m{str(self.mutation)}_r{str(self.ratio)}'
        
class Data_props:
    def __init__(self,eligible =[], bulk = [], free_spots = [], n0 = [], n1 =[], spot0 = {}, spot1 = {}, bulk0 = [], bulk1 = [], props_detach = [], props_attach = [],d_props_daughter = {}):
        self.eligible = eligible
        self.bulk = bulk
        self.free_spots = free_spots
        self.n0 = n0
        self.n1 = n1
        self.spot0 = spot0
        self.spot1 = spot1
        self.bulk0 = bulk0
        self.bulk1 = bulk1
        self.props_detach = props_detach
        self.props_attach = props_attach
        self.props_daughter = d_props_daughter
        
    def __repr__(self):
 
        return f'eligible: {self.eligible}, \n bulk {self.bulk} \n free spots {self.free_spots} \n n0 {self.n0} \n n1 {self.n1} \n spot0 {self.spot0} \n spot1 {self.spot1} \n bulk0 {self.bulk0} \n bulk1 {self.bulk1} \n props detach {self.props_detach} \n props daughter {self.props_daughter}'
        

def propensities_templating(surface,solution,choice):
    
    monos = []; bulk = []
    spots = {}
    
    for species in solution:
        if species.sequence == choice:
            monos.append(species)
        else:
            bulk.append(species)
            
    for i in range(len(surface)):
        if 'x' in surface[i].daughter:
            for y in range(len(surface[i].daughter)):
                if surface[i].daughter[y] == 'x' and surface[i].sequence[y] == choice:
                    spots = populate_dict_list(i, y, spots)            
    
    return monos, spots, bulk

def template(mono,spots,surface):
    
    probs = []
    l_spots = []
    #makes 2 lists based on the dictionary spots. spots is in the format {index of surface:index of daughter}
    for k,v in spots.items():
        l_spots.append(k)
        probs.append(len(v))
    rel_probs = [i/sum(probs) for i in probs]
    #picks an index of the list l_spots based on the length of daughter spots 
    index = np.random.choice(range(len(l_spots)), p = rel_probs, replace=False)
    #choice is itself an index for the list surface
    choice_pol = l_spots[index]
    daughter_index = random.randrange(len(spots[choice_pol]))
    choice_daughter = spots[choice_pol][daughter_index]
    species = surface[choice_pol]
    species.daughter = replace_at_index(species.daughter, choice_daughter, mono.sequence)
    
    return surface

def replace_at_index(seq, index, char):
    if index < len(seq):
        new_seq = seq[0:index] + char + seq[index+1:]
    else:
        print('Error in replacing')
    return new_seq

def new_daughter_seq(l, index):
    new_seq = ''
    for i in range(len(l)):
        if i == index:
            new_seq += 'x'*len(l[i])
        else:
            new_seq += l[i]
    
    return new_seq

def len_dict(d):
    num = 0
    for k,v in d.items():
        num += len(v)
    return num

def calculate_props(choice,run,data_props,solution,surface,templating,selection):
    d_ = {}
#     print('recalculating props')
#     print(choice)
    
    if choice == 'mono':
        ##propensity that a monomer flows in
        prop = run.rates['flow'] * run.m0
    
    elif choice == 'flow':
        ##propensity that a species flows out
        prop = run.rates['flow'] * len(solution)
    
    elif choice == 'surface':
        ##propensity that a species attaches to surface
        #calculate how many monomers compose all the species attached to the surface
        vol = 0
        for species in surface:
            vol += len(species.sequence)
        #calculate how many free spots there are
        data_props.free_spots = run.spots - vol
#         print(f'in surface props, vol is {vol}, free spots is {data_props.free_spots}')
        if data_props.free_spots < 0:
            print('Error in free spots')
        #identify species in solution that could attach to those free spots
        data_props.eligible = []
        data_props.bulk = []
        data_props.props_attach = []
    
        for species in solution:
            if len(species.sequence) <= data_props.free_spots:
                data_props.eligible.append(species)
                spots = 1 + data_props.free_spots - len(species.sequence)
                data_props.props_attach.append(run.rates['surface']*spots)
                
            else:
                data_props.bulk.append(species)
#         print(f'eligible {data_props.eligible}, bulk is {data_props.bulk}')
        prop = sum(data_props.props_attach)
        
    elif choice == 't0':
        ##propensity that templation happens
        if templating:
            data_props.n0,data_props.spot0,data_props.bulk0 = propensities_templating(surface,solution, '0')
#             print(f'in t0 props: n0 {data_props.n0}, spot0 {data_props.spot0}, bulk0 {data_props.bulk0}')
              
            prop = run.rates['template'] * len(data_props.n0) * len_dict(data_props.spot0)
        else:
            prop = 0
            
    elif choice == 't1':
        ##propensity that templation happens
        if templating:
            data_props.n1,data_props.spot1,data_props.bulk1 = propensities_templating(surface,solution, '1')
#             print(f'in t1 props: n1 {data_props.n1}, spot1 {data_props.spot1}, bulk1 {data_props.bulk1}')
            prop = run.rates['template'] * len(data_props.n1) * len_dict(data_props.spot1)
        else:
            prop = 0
            
    elif choice == 'detach':
        ##propensity that species detached from the surface
        #calculate fitness and probability of detachment (inverse fitness)
        fitness = calc_fitness(surface, run.fitnessform)
#         print('fitness')
#         print(fitness)
#         print(f'detach props: fitness {fitness}')
        if selection:
            probs_det = calc_probs(fitness)
        else:
            probs_det = [1 for i in range(len(surface))]
#         print(f'probs_det {probs_det}')    
        #calculate prop_detach
        data_props.props_detach = []
#         print('probs det then rate detach')
#         print(probs_det)
#         print(run.rates['detach'])
        for species_prob in probs_det:
            data_props.props_detach.append(run.rates['detach']*species_prob)
#         print(f'props_detach {data_props.props_detach}')
        prop = sum(data_props.props_detach)

    elif choice == 'daughter':
        ##propensity that daughter species detached from parent
        #get list of daughters
        d_daughters = get_daughters(surface)
        #calculate prop_daughter
        data_props.props_daughter = {}
        prop = 0
        for k,v in d_daughters.items():
            probs = []
            for daughter in v:
                if 'x' in daughter: 
                    probs.append(0)
                else: 
                    prob = run.rates['daughter']**len(daughter)
                    probs.append(prob)
                    prop += prob
            data_props.props_daughter[k] = probs
        d_ = d_daughters
#         print(f'daughter props: props_daughter {data_props.props_daughter}')
    return prop,run,data_props,solution,surface,d_

def get_daughters(surface):
    d_daughters = {}
#     print('getting daughters')
    for i in range(len(surface)):
        l = []
#         print(surface[i])
        word = surface[i].daughter[0]
        if len(surface[i].daughter) == 1: l.append(word)
        for char in range(1,len(surface[i].daughter)):
            if (surface[i].daughter[char] == 'x'and word[-1] == 'x') or (surface[i].daughter[char] != 'x'and word[-1] != 'x'):
                word += surface[i].daughter[char]
            else:
                l.append(word)
                word = surface[i].daughter[char] 
            if char == len(surface[i].daughter)-1:
                l.append(word)
        if l != []:
            d_daughters[i] = l
    return d_daughters







def model(gen,previous_surface,previous_solution,templation,selection,run,props,data_props, csvbool):
    """
    TO EDIT
    Takes in thirteen variables: form (string or list), gen (int), previous_gen (list), max_gen (int), pool_size (int),
    first (bool), template (bool), selection (bool), step (int), mutation (float), prob_1 (float), initial_pool (int), packed_dicts (tuple).
    Initializes the generation by unpacks the dicts in packed_dicts, creating a new list 'new_species'.
    Elongates the species in l_species_ID, generates a pool of monomers of size 'pool_size' by calling 'generate_monos'.
    If template is True then templated the polymers in new_species by calling 'replicate' then adds them to the pool of polymers
    along with the leftover monomers (else just adds the monomers). Calculates 'fitness' and 'probs' of dying then kills
    species in order to return to the size 'initial_pool', based on 'probs'. Saves information of the species at the
    end of the gen in 'd_run', of the ones killed in 'd_killed' and information on partial templation in 'd_partial'.
    If the model didn't reach the last generation then it iterates.
    At the end it returns three variables: d_run (dict), d_killed (dict), d_partial (dict).
    """

    d_detached = {}
    events = {'detachment':0,'templation':0,'daughter':0,'attachment':0,'flow':0,'mono':0}

    surface = copy.deepcopy(previous_surface)
    solution = copy.deepcopy(previous_solution)

    running = True
    counter = 1
    while running:

        new_props = []
        event = ''
        for prop in props:
            if type(prop) is str:
                new_prop,run,data_props,solution,surface,d_ = calculate_props(prop,run,data_props,solution,surface,templation,selection)
                if prop == 'daughter': d_daughters = d_
                new_props.append(new_prop)
            else:
                new_props.append(prop)
                
        prop_mono,prop_flow,prop_surf,prop_0,prop_1,prop_detach,prop_daughter = new_props

        total = sum(new_props)
         
        #add to time    
        tau = np.random.exponential(scale=1/total) 
        current_time = run.time[-1] + tau
        run.time.append(current_time)
        
        ###pick event
        #pick random number between 0 (inclusive) and 1 (exclusive)
        rand_num = np.random.uniform()
        #weight that number by the propesnity total
        num = rand_num*total
        
        #pick monomer flows in
        if num < sum(new_props[:1]):
#             print('>>flow in')
            if np.random.uniform() < run.ratio:
                seq = '1' 
            else:
                seq = '0'
            mono = Polymer(seq, gen) 
            solution.append(mono)
            events['mono'] +=1
            event = 'mono'
            gen +=1

            if gen%run.save_point == 0 and counter != 0: running = False
            counter+=1
        
        #pick species flows out
        elif sum(new_props[:1]) <= num and num < sum(new_props[:2]):
#             print('>>flow out')
            choice = random.randrange(len(solution))
            flowout = solution.pop(choice)
            events['flow'] +=1
            event = 'flow'
            run.killed[current_time] = flowout
            if '0' in flowout.daughter or '1' in flowout.daughter:
                print('\n XXXXXXXX ERROR XXXXX')
        
        #pick species attaches to surface
        elif sum(new_props[:2]) <= num and num < sum(new_props[:3]):

            rel_probs = [i/prop_surf for i in data_props.props_attach]

            choice = np.random.choice(range(len(data_props.eligible)), p = rel_probs, replace=False)
            species = data_props.eligible.pop(choice)
            
            event = 'attachment'
            if len(species.sequence) == 1:
                event += ' mono'
                #calculate number of spots that corresponds to ends of species on surface
                species.daughter = 'x'
                end_spots = len(surface)*2
                number = random.randrange(data_props.free_spots)
                if number > 0 and number <= end_spots:
                    event += ' elongation'
                    pick = surface[random.randrange(len(surface))]
                    if np.random.uniform() < 0.5:
                        pick.sequence += species.sequence
                        pick.daughter += species.daughter
                    else:
                        pick.sequence = species.sequence + pick.sequence
                        pick.daughter = species.daughter + pick.daughter
                else:
                    surface.append(species)
            else:
                surface.append(species)
            solution = data_props.bulk + data_props.eligible
            events['attachment'] +=1
            
            
        #pick 0 templates
        elif sum(new_props[:3]) <= num and num < sum(new_props[:4]):

            choice = random.randrange(len(data_props.n0))
            mono = data_props.n0.pop(choice)
            surface = template(mono,data_props.spot0,surface)
            solution = data_props.n0 +  data_props.bulk0
            events['templation']+=1
            event = 'templation 0'
        
        #pick 1 templates
        elif sum(new_props[:4]) <= num and num < sum(new_props[:5]):

            choice = random.randrange(len(data_props.n1))
            mono = data_props.n1.pop(choice)
            surface = template(mono,data_props.spot1,surface)
            solution = data_props.n1 + data_props.bulk1
            events['templation']+=1
            event = 'templation 1'
            
        #pick species detaches from surface
        elif sum(new_props[:5]) <= num and num < sum(new_props[:6]):
#             print('>>detach')
            rel_probs = [i/prop_detach for i in data_props.props_detach]
            choice = np.random.choice(range(len(surface)), p = rel_probs, replace=False)
            detach = surface.pop(choice)
            if choice in list(d_daughters.keys()):
                for seq in d_daughters[choice]:
                    if 'x' not in seq: 
                        daughter = Polymer(seq, gen, parent = detach, daughter = 'x'*len(seq)) 
                        solution.append(daughter)
                detach.daughter = 'x'*len(detach.sequence)
            solution.append(detach)
            events['detachment']+=1
            event = 'detachment'
         
        #pick daughter detaches from parent
        elif sum(new_props[:6]) <= num and num < total:
#             print('>>daughter')
            rel_probs = []
            l_spots = []
            #makes 2 lists based on the dictionary data_props.props_daughters. 
            #data_props.props_daughters is in the format {index of surface:list of probs of detach}
            for k,v in data_props.props_daughter.items():
                for i in range(len(v)):
                    #creates a coordinate for the dict data_props.props_daughters
                    l_spots.append([k,i])
                    rel_probs.append(v[i]/prop_daughter)
            #picks an index of the list l_spots based on the length of daughter spots 
            choice = np.random.choice(range(len(rel_probs)), p = rel_probs, replace=False)
            #unpacking the coordinates
            choice_pol,daughter_index = l_spots[choice]
            #modify the daughter sequence of the parent polymer to replace the leaving daughter with xs
            species = surface[choice_pol]
            species.daughter = new_daughter_seq(d_daughters[choice_pol], daughter_index)
            #detach daughter sequence and add to solution
            seq = d_daughters[choice_pol][daughter_index]
            daughter = Polymer(seq, gen, parent = species, daughter = 'x'*len(seq))
            solution.append(daughter)
            
            events['daughter']+=1
            event = 'daughter'
            
        else:
            print('Error in picking event')
        
        # if csvbool:
        #     row = [current_time,prop_mono,prop_flow,prop_surf,prop_0,prop_1,prop_detach,prop_daughter,len(solution),len(surface),data_props.free_spots,running]    
        #     with open(run.name +'.csv', 'a+', newline ='') as csvfile: 
        #         csvwriter = writer(csvfile)
        #         csvwriter.writerow(row)
            
            
        props = [prop_mono,'flow','surface','t0','t1','detach','daughter']

    details_sol = {};details_surf = {}
    for species in solution:
        details_sol = populate_dict_list(len(species.sequence), species, details_sol)
    run.solution[current_time] = details_sol
    for species in surface:
        details_surf = populate_dict_list(len(species.sequence), species, details_surf)
    run.surface[current_time] = details_surf
    run.events[current_time] = copy.deepcopy(events)
    
    c_surface = copy.deepcopy(surface)
    c_solution = copy.deepcopy(solution)
    props[-1] = 'daughter'

    if gen >= run.maxgen:
        print('finished')
        return run
    return model(gen,c_surface,c_solution,template,selection,run,props,data_props, csvbool)


def run_model(tup, duplicates, form_fitness, save_point, save = False, csvbool=False, i=0):
    """
    TO EDIT
    Takes in four variables: tup (tuple), duplicates (int), form (string or list), save (bool, default = False).
    Unpackes the variables in 'tup'. Runs the model X times (X = duplicates) based on the variables stored in 'tup'.
    For each run, initiates the model by creating storage variables d_run, d_killed and d_partial (dicts).
    Makes a pool of initial monomers of size 'initial_pool'.
    Stores the details of the initial species in the dict 'd_run' and will do so for each generation.
    Calls 'model' and runs for X gens (X = maxg). Creates a Run that saves all the variables and the data from that run.
    If save = True, saves the run as a pickle (see path).
    Returns one variable: current_run (Run).
    """
    all_runs = []
    # print('\n NEW RUN')
    
    # print(tup)
    # print(form_fitness)
    date,t,sel,spots,pool_size,rates_og,fitness,mutation,ratio,rate_name,m0 = tup

    rates = copy.deepcopy(rates_og)
    # print(rates)
    template, selection = (False, False)
    if t == 't':
        template = True
    else: rates['template'] = 0
    if sel == 'sel':
        selection = True
    # print(rates)
    for i in range(1,duplicates+1):
        ##### #VARIABLES
        max_gen = int(pool_size)
        
        #INTITALIZATION
        d_killed, d_events, d_solution, d_surface = [{} for i in range(4)]
        time = [0]
        
        gen = 0
        details = {1:[]}
        d_solution[time[0]] = details
            
        d_surface[time[0]] = details
        d_events[time[0]] = {'detachment':0,'templation':0,'daughter':0,'attachment':0,'flow':0,'mono':0}
        
        #make a Class Run object
        current_run = Run(date,t,sel,spots,save_point,pool_size,rates,fitness,form_fitness,mutation,ratio,rate_name,m0,d_solution, d_surface, d_killed, d_events, time, max_gen, 'na')
        props = ['mono','flow','surface','t0','t1','detach','daughter']
        data_props = Data_props()
        print(current_run)

        # field names 
        fields = ['time','prop_mono','prop_flow','prop_surf','prop_0','prop_1','prop_detach','prop_daughter','sol','surf','vol surf', 'save'] 

        # # name of csv file 
        # filename = f"{str(current_run)}_{i}"
        # current_run.name = filename
        # if csvbool:
        # # writing to csv file 
        #     with open(filename + '.csv', 'w') as csvfile: 
        #         # creating a csv writer object 
        #         csvwriter = csv.writer(csvfile) 

        #         # writing the fields 
        #         csvwriter.writerow(fields) 
            
        #run model
        current_run = model(gen,[],[],template,selection,current_run,props,data_props, csvbool)
        print('\n ')
        
        new_name = f"/users/natalieg/{filename}.p"

        # print(filename)
        # print(new_name)
        # print(i)
        i += 1
        
        if save:
            print('saving')
            pickle.dump(current_run, open(new_name, "wb"))
        all_runs.append(current_run)
    print('END RUN')
    return all_runs


df = pd.read_csv('Feb10combinations.csv')
date = 'May23'
save_point = 100
dup = 4

for i in range(len(df)):
    row = df.iloc[i]
    cs = row['combo']
#     print(row)
    spots = row['spots']
    pool_size = row['pool_size']
    m0 = row['m0']
    rates = {'flow': row['flow'],
             'surface': row['surface'], 
             'template': row['template'], 
             'detach': row['detach'], 
             'daughter': row['daughter']}
    fitnessform = row['fitness_eq']
    alpha = row['alpha']
    fitnessname = fitnessform + str(alpha)

    run_model((date, 't', 'nosel', spots, pool_size, rates, fitnessname, 0, 0.5, cs, m0), dup, [fitnessform, alpha], save_point, True, False)
    run_model((date, 't', 'sel', spots, pool_size, rates, fitnessname, 0, 0.5, cs, m0), dup, [fitnessform, alpha], save_point, True, False)




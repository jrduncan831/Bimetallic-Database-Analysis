import numpy as np 
import matplotlib.pyplot as plt

Code_to_text = {'H':"HER",'O':"ORR","S":"Slab","N":"Nanoparticle"}
target_BE = {'O':-1.143,'H':-0.4915} # Ideal binding energies for ORR ('O')and HER ('H'

# Get Materials Key Functions
def get_keys_for_material(df,metalA,metalB,material_type,reaction,ensemble):
    """
    Returns dataframe key based on descriptors given as input variables
    """
    if metalA == metalB:
        key = material_type +'/'+ metalA + '/None/'+metalB+'/None/' + reaction
        return key
    else:
        if ensemble == 'AAA':
            key1 = material_type +'/'+metalA+'/3/'+metalB+'/0/' + reaction
            key2 = material_type +'/'+metalB+'/0/'+metalA+'/3/' + reaction
        else:
            key1 = material_type +'/'+metalA+'/2/'+metalB+'/1/' + reaction
            key2 = material_type +'/'+metalB+'/1/'+metalA+'/2/' + reaction
    if key1 in df.index:
        return key1
    elif key2 in df.index:
        return key2
    else:
        return None 

def get_pure_key(metal, material_type, reaction):
    '''
    Returns pure metal binding energy key based on descriptors
    '''
    return material_type +'/'+metal+'/None/'+metal+'/None/' + reaction

def print_material_from_key(key):
    '''
    Prints readable text describing material based on key
    '''
    key_split = key.split('/')
    print(key_split[1] + key_split[2] + key_split[3] + key_split[4] +' '+Code_to_text[key_split[0]] + ' for ' + Code_to_text[key_split[5]])

# Plot binding energy versus ensemble 

def get_BE_AAA_and_AAB(df, metalA,metalB,material_type,reaction):
    """
    Returns binding energy + std for AAA and AAB ensemble for metal combination
    """
    ensemble_array,std_array = np.zeros(2),np.zeros(2)
    for i,ensemble in enumerate(['AAA','AAB']):
        key = get_keys_for_material(df, metalA,metalB,material_type,reaction,ensemble)
        if key != None:
            ensemble_array[i],std_array[i] = df.loc[key,'result'],df.loc[key,'std']
        else:
            return None,None 
    return ensemble_array,std_array 

    

def get_BE_with_changing_ensemble(df, metalA,metalB,material_type,reaction):
    """
    Returns binding energy array for four possible ensembles and cooresponding standard deviation
    """
    be_array,std_array = np.zeros(4),np.zeros(4)
    be_array[:2], std_array[:2] = get_BE_AAA_and_AAB(df, metalA,metalB,material_type,reaction)
    be_array[2:], std_array[2:] = get_BE_AAA_and_AAB(df, metalB,metalA,material_type,reaction)
    be_array[2],be_array[3] = be_array[3],be_array[2]
    std_array[2],std_array[3] = std_array[3],std_array[2]
    return be_array, std_array

def plot_BE_change_with_ensemble(df, metalA,metalB,material_type,reaction,show_pure=False):
    """
    Parameters:
    show_pure - Set to True if you want to show pure binding energies of two metals
    """
    num_metal_A = np.linspace(0,3,4)
    fig, ax = plt.subplots()
    be_array, std_array = get_BE_with_changing_ensemble(df, metalA,metalB,material_type,reaction)
    ax.scatter(num_metal_A,be_array,label= metalA + metalB + ' binding energy')
    ax.errorbar(num_metal_A,be_array,std_array)
    if show_pure:
        metalA_be = df.loc[get_pure_key(metalA, material_type, reaction),'result']
        metalB_be = df.loc[get_pure_key(metalB, material_type, reaction),'result']
        ax.plot(num_metal_A,metalA_be * np.ones(len(num_metal_A)),label=metalA + ' binding energy')
        ax.plot(num_metal_A,metalB_be * np.ones(len(num_metal_A)),label=metalB + ' binding energy')
    ax.legend()
    ax.set_xticks(num_metal_A)
    ax.set_xticklabels([3*metalA,(2*metalA)+metalB, metalA+(2*metalB),3*metalB])
    ax.set_xlabel('Ensemble')
    ax.set_ylabel('Average Binding Energy (eV)')
    ax.set_title("Binding Energy versus Ensemble of "+ metalA + metalB + ' '+ Code_to_text[material_type] +" for the "+Code_to_text[reaction])

# Get catalytic activity from binding energy

def get_activity_ORR(binding_energy):
    """
    Parameters:
    binding_energy - oxygen binding energy
    """
    if binding_energy < target_BE['O']:
        return 0.7642240985316116 + 0.7854675573688066 * binding_energy
    else:
        return -3.001067465601418 - 2.507767421216152 * binding_energy
    
def get_activity_HER(binding_energy):
    """
    Parameters:
    binding_energy - hydrogen binding energy
    """
    if float(binding_energy)  < target_BE['H']:
        return 16.738 * (float(binding_energy) + 0.4915) - 1.6802
    else:
        return -16.592 * (float(binding_energy) + 0.4915) - 1.7559

def get_average_catalytic_ability_HER(binding_energy_list):
    """
    Parameters:
    binding_energy_list - array of 5 hydrogen binding energies from randomly generated structures 
    """
    binding_energy_list = np.array([float(i) for i in binding_energy_list])
    be_array = list(map(get_activity_HER, np.array(binding_energy_list)))
    return np.mean(be_array)

def get_average_catalytic_ability_ORR(binding_energy_list):
    """
    Parameters:
    binding_energy_list - array of 5 oxygen binding energies from randomly generated structures 
    """
    binding_energy_list = np.array([float(i) for i in binding_energy_list])
    be_array = list(map(get_activity_ORR, np.array(binding_energy_list)))
    return np.mean(be_array)

# Get catalytic activity from binding energy

def get_catalytic_actvity_matrix(df, metalA,metalB,material_type,reaction,ensemble):
    """
    Returns catalytic activity matrix for heat map 
    """
    binding_energy_AAA = np.zeros((len(metalA),len(metalB)))
    over_or_under_binding = np.zeros((len(metalA),len(metalB)))
    for i,m1 in enumerate(metalA):
        for j,m2 in enumerate(metalB):
            key = get_keys_for_material(df,m1,m2,material_type,reaction,ensemble)
            if key != None:
                binding_energy_AAA[j][i] = df.loc[key,'catalytic_ability']
                over_or_under_binding[j][i] = df.loc[key,'result']<target_BE[reaction]
            else:
                binding_energy_AAA[j][i] = np.nan
                over_or_under_binding[j][i] = np.nan
    return binding_energy_AAA, over_or_under_binding

def plot_catalytic_activity(df, metalA,metalB,material_type,reaction,ensemble,show_catalytic_activity=False):
    """
    Returns heat map of the predicted catalytic ability for listed metals and fixe.
    Red is underbinding activity, blue is underbinding,and black indicates no data
    
    Parameters:
    metalA - list of metals shown in heat map
    metalB - list of metals shown in heat map
    show_catalytic_activity - if True will show numerical values for catalytic ability
    """
    binding_energy_AAA, under_binding = get_catalytic_actvity_matrix(df,metalA,metalB,material_type,reaction,ensemble)
    fig, ax = plt.subplots(dpi=120)
    # Adds a black square when there is not existing data
    im0 = ax.imshow(np.nan_to_num(under_binding - 2),cmap=plt.cm.flag)
    
    # Adds a red squares when under binding; 1.7 choosen such that Nan values do not disrupt range of contour plots
    im1 = ax.imshow(np.ma.masked_array(np.nan_to_num(binding_energy_AAA),
                                        mask=np.nan_to_num(under_binding-1)+1),
                                        cmap=plt.cm.Reds,
                                        vmin=np.amin(np.nan_to_num(binding_energy_AAA+1.7))-1.7,
                                        vmax =np.amax(np.nan_to_num(binding_energy_AAA+1.7))-1.7)
    cba = plt.colorbar(im1)
    cba.set_ticks([np.amin(np.nan_to_num(binding_energy_AAA)),df.loc['S/Pt/None/Pt/None/'+reaction,'catalytic_ability']])
    cba.set_ticklabels(['low','Pt'])
    cba.set_label('Activity When Under Binding')

    # Adds blue squares when under binding -- switch is to flip the masking array 
    switch = lambda x: (x+1)%2
    im2 = ax.imshow(np.ma.masked_array(np.nan_to_num(binding_energy_AAA),
                                        mask=np.nan_to_num(np.array(list(map(switch,under_binding)))-1)+1),
                                        cmap=plt.cm.Blues,
                                        vmin=np.amin(np.nan_to_num(binding_energy_AAA+1.7))-1.7,
                                        vmax = np.amax(np.nan_to_num(binding_energy_AAA+1.7))-1.7
                                       )
    cba = plt.colorbar(im2)
    cba.set_ticks([np.amin(np.nan_to_num(binding_energy_AAA)),df.loc['S/Pt/None/Pt/None/'+reaction,'catalytic_ability']])
    cba.set_ticklabels(['low','Pt'])
    cba.set_label('Actvity When Over Binding')
    ax.set_xticks(np.arange(len(metalA)))
    ax.set_yticks(np.arange(len(metalB)))
    ax.set_xticklabels(metalA)
    ax.set_yticklabels(metalB)
    ax.set_xlabel('Metal A')
    ax.set_ylabel('Metal B')
   # print('feedback:',ax.get_xlim())
   # ax.set_xlim([-1.0,8.0]) 
   # print('feedback:',ax.get_xlim())
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
    if show_catalytic_activity == True:
        for i in range(len(metalA)):
            for j in range(len(metalB)):
                text = ax.text(j, i, round(binding_energy_AAA[i, j],1),
                       ha="center", va="center", color="w")
    ax.set_title("Cataltyic Ability of "+ Code_to_text[material_type] +"\n for "+ensemble+ " site for the "+ Code_to_text[reaction] +" reaction")

 

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



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

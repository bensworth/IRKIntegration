# List all details about IRK schemes to assist with plotting.

# IRK families
def Families():
    families = {
        -1: "ASDIRK",
        0 : "LSDIRK", 
        1 : "Gauss",
        2 : "Radau\,IIA",
        3 : "Lobatto\,IIIC"}
    return families
    
# IRK types
def Labels():
    labels = {
        -1 : "A\\rm{-}SDIRK",
        0 : "L\\rm{-}SDIRK", 
        1 : "Gauss",
        2 : "Radau\,IIA",
        3 : "Lobatto\,IIIC"}
    return labels        

# Orders
def Orders():
    orders = {  
        -13 : 3, -14 : 4,
        1 : 1, 2 : 2, 3 : 3, 4 : 4,
        12 : 2, 14 : 4, 16 : 6, 18 : 8, 110 : 10,
        23 : 3, 25 : 5, 27 : 7, 29 : 9, 
        32 : 2, 34 : 4, 36 : 6, 38 : 8}
    return orders
    
# Stages
def Stages():
    stages = {  
        -13 : 2, -14 : 3,
        1 : 1, 2 : 2, 3 : 3, 4 : 5,
        12 : 1, 14 : 2, 16 : 3, 18 : 4, 110 : 5,
        23 : 2, 25 : 3, 27 : 4, 29 : 5, 
        32 : 2, 34 : 3, 36 : 4, 38 : 5}
    return stages    

# Labels of individual schemes    
def IndividualLabels():    
    labels = { 
        -13 : "A\\rm{-}SDIRK(3)", -14 : "A\\rm{-}SDIRK(4)",
        1 : "L\\rm{-}SDIRK(1)", 2 : "L\\rm{-}SDIRK(2)", 3 : "L\\rm{-}SDIRK(3)", 4 : "L\\rm{-}SDIRK(4)",
        12 : "Gauss(2)", 14 : "Gauss(4)", 16 : "Gauss(6)", 18 : "Gauss(8)", 110 : "Gauss(10)",
        23 : "Radau\, IIA(3)", 25 : "Radau\, IIA(5)", 27 : "Radau\, IIA(7)", 29 : "Radau\, IIA(9)",
        32 : "Lobatto\, IIIC(2)", 34 : "Lobatto\, IIIC(4)", 36 : "Lobatto\, IIIC(6)", 38 : "Lobatto\, IIIC(8)"} 
    return labels                 
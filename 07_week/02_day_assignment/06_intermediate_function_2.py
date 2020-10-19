#============================================
# Arwa Ashi - HW 2 - Week 7 - Oct 19, 2020
#============================================

# 1 # ===========================================
# Update Values in Dictionaries and Lists
# ===============================================
x = [ [5,2,3], [10,8,9] ] 
students_01 = [
     {'first_name':  'Michael', 'last_name' : 'Jordan'},
     {'first_name' : 'John', 'last_name' : 'Rosales'}
]
sports_directory = {
    'basketball' : ['Kobe', 'Jordan', 'James', 'Curry'],
    'soccer' : ['Messi', 'Ronaldo', 'Rooney']
}
z = [ {'x': 10, 'y': 20} ]

#Change the value 10 in x to 15. Once you're done, x should now be [ [5,2,3], [15,8,9] ].
x[1][0] = 15; print(x)

#Change the last_name of the first student from 'Jordan' to 'Bryant'
students_01[0]['last_name'] = 'Bryant'; print(students_01[0])

#In the sports_directory, change 'Messi' to 'Andres'
sports_directory['soccer'][0] = 'Andres'; print(sports_directory['soccer'])

#Change the value 20 in z to 30
z[0]['y'] = 30; print(z[0])


# 2 # ===========================================
# Iterate Through a List of Dictionaries
# Create a function iterateDictionary(some_list) that, given a list of dictionaries,
# the function loops through each dictionary in the list and prints each key and the associated value.
# ===============================================
students = [
         {'first_name' : 'Arwa   ', 'last_name' : 'Ashi   ', 'city':'Jeddah', 'country':'Saudi Arabia'},
         {'first_name' : 'Michael', 'last_name' : 'Jordan '},
         {'first_name' : 'John   ', 'last_name' : 'Rosales'},
         {'first_name' : 'Mark   ', 'last_name' : 'Guillen'},
         {'first_name' : 'KB     ', 'last_name' : 'Tonel  '}
    ]

def iterateDictionary(func_list):
    func_list         = func_list
    for j in range(len(func_list)):
        dict_keys      = func_list[j].keys()
        dict_keys_list = []
        for key in dict_keys:
            dict_keys_list.append(key)
        for i in range(len(dict_keys_list)):
            print("{} - {}, ".format(dict_keys_list[i],func_list[j][dict_keys_list[i]]),end = '')
        print()#(dict_keys_list_01)

test_Iter_dict_01 = iterateDictionary(students)


# 3 # ===========================================
# Get Values From a List of Dictionaries
# Create a function iterateDictionary2(key_name, some_list) that,
# given a list of dictionaries and a key name, the function prints the value stored
# in that key for each dictionary.
# ===============================================
def iterateDictionary2(key_name, func_list):
    func_list         = func_list
    for i in range(len(func_list)):
        if key_name in func_list[i].keys():
            print("{}".format(func_list[i][key_name]))
        else:
            print("please update the {} for others !".format(key_name,func_list))

test_Iter_dict_02 = iterateDictionary2('first_name', students)
test_Iter_dict_03 = iterateDictionary2('last_name', students)
test_Iter_dict_03 = iterateDictionary2('city', students)
test_Iter_dict_03 = iterateDictionary2('country', students)


# 4 # ===========================================
# Iterate Through a Dictionary with List Values
# Create a function printInfo(some_dict) that given a dictionary whose values are all lists, prints the name of each
# key along with the size of its list, and then prints the associated values within each key's list.
# ===============================================
dojo = {
   'locations'  : ['San Jose', 'Seattle', 'Dallas', 'Chicago', 'Tulsa', 'DC', 'Burbank'],
   'instructors': ['Michael', 'Amy', 'Eduardo', 'Josh', 'Graham', 'Patrick', 'Minh', 'Devon']
}

print(dojo['locations'][0])

def printInfo(func_dict):
    func_dict           = func_dict
    func_dict_keys      = func_dict.keys()
    func_dict_keys_list = []
    for key in func_dict.keys():
        func_dict_keys_list.append(key)
    for i in range(len(func_dict_keys_list)):
        print("{} {}".format(len(func_dict[func_dict_keys_list[i]]),func_dict_keys_list[i].upper()))
        for j in range(len(func_dict[func_dict_keys_list[i]])):
            print(func_dict[func_dict_keys_list[i]][j])

test_PI_01 = printInfo(dojo)
test_PI_02 = printInfo(sports_directory)




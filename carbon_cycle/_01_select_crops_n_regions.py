#!/usr/bin/env python

import sys, os
import numpy as np

# This script maps NUTS ids to EUROSTAT region names, and crop ids to crop names 

#===============================================================================
def main():
#===============================================================================
    """
	This scripts constructs a dictionary of NUTS_ids <--> NUTS names, of
    crop_ids <--> crop names, and a list of years, which are all short-selected 
    by the user, and stores them in pickle files in the current directory.

    SELECTION OF NUTS REGIONS:
    --------------------------

    1: we read the NUTS id codes from the shapefile we use to plot NUTS regions
       we select only the codes that correspond to the NUTS levels we are
       interested in (NUTS 1 or 2, it varies per country).
       ==> we obtain a shortlist of NUTS 1 and 2 id codes we want to do 
       simulations for.

    2: we assign a corrected NUTS name (no accents, no weird alphabets), and a
       latin-1 encoded NUTS name (same encoding as the EUROSTAT observation 
       files) to each NUTS code of the dictionary.

    SELECTION OF CROPS:
    -------------------

    3: we simply manually map the EUROSTAT crop names to each crop code

    SELECTION OF YEARS:
    -------------------

    4: we create a list of (integer) years

    STORING IN MEMORY:
    ------------------

    4: we pickle the produced dictionaries/lists and store them in the 
       current directory

    """
#-------------------------------------------------------------------------------
    from cPickle import load as pickle_load
    from cPickle import dump as pickle_dump
#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================

    # for each country code, we say which NUTS region level we want to 
    # consider (for some countries: level 1, for some others: level 2)
    lands_levels = {'AT':1,'BE':1,'BG':2,'CH':1,'CY':2,'CZ':1,'DE':1,'DK':2,
                    'EE':2,'EL':2,'ES':2,'FI':2,'FR':2,'HR':2,'HU':2,'IE':2,
                    'IS':2,'IT':2,'LI':1,'LT':2,'LU':1,'LV':2,'ME':1,'MK':2,
                    'MT':1,'NL':1,'NO':2,'PL':2,'PT':2,'RO':2,'SE':2,'SI':2,
                    'SK':2,'TR':2,'UK':1}

    # list of selected crops of interest:
    crops = ['Potato']#,'Spring wheat','Winter wheat',
            # 'Spring barley','Winter barley','Spring rapeseed','Winter rapeseed',
            # 'Rye','Potato','Sugar beet','Sunflower','Field beans']

    # list of selected years to simulate the c cycle for:
    years = [2006]#,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]

    # If you want to check if we match the crop and region names in the EUROSTAT
    # files, set the following files to true, it will print the result to screen
    check_eurostat_file = False

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    currentdir    = os.getcwd()
    EUROSTATdir   = 'EUROSTATobs/'
#-------------------------------------------------------------------------------
# create a temporary directory if it doesn't exist
    if not os.path.exists("../tmp"):
        os.makedirs("../tmp")

#-------------------------------------------------------------------------------
# 1- WE CREATE A DICTIONARY OF REGIONS IDS AND NAMES TO LOOP OVER
#-------------------------------------------------------------------------------
# Select the regions ID in which we are interested in
    NUTS_regions = make_NUTS_composite(lands_levels, EUROSTATdir)
#-------------------------------------------------------------------------------
# Create a dictionary of NUTS region names, corresponding to the selected NUTS id
    NUTS_names_dict = map_NUTS_id_to_NUTS_name(NUTS_regions, EUROSTATdir)
#-------------------------------------------------------------------------------
# pickle the produced dictionary in the current directory:
    pathname = os.path.join(currentdir, '../tmp/selected_NUTS_regions.pickle')
    if os.path.exists(pathname): os.remove(pathname)
    pickle_dump(NUTS_names_dict, open(pathname,'wb'))

#-------------------------------------------------------------------------------
# 2- WE CREATE A DICTIONARY OF CROP IDS AND NAMES TO LOOP OVER
#-------------------------------------------------------------------------------
# Create a dictionary of crop EUROSTAT names, corresponding to the selected crops
    crop_names_dict = map_crop_id_to_crop_name(crops)
#-------------------------------------------------------------------------------
# pickle the produced dictionary:
    pathname = os.path.join(currentdir, '../tmp/selected_crops.pickle')
    if os.path.exists(pathname): os.remove(pathname)
    pickle_dump(crop_names_dict, open(pathname,'wb'))

#-------------------------------------------------------------------------------
# 3- WE CREATE A LIST OF YEARS TO LOOP OVER?
#-------------------------------------------------------------------------------
# pickle the produced dictionary:
    print '\nWe select the following years:\n', years
    pathname = os.path.join(currentdir, '../tmp/selected_years.pickle')
    if os.path.exists(pathname): os.remove(pathname)
    pickle_dump(years, open(pathname,'wb'))

#-------------------------------------------------------------------------------
# 4- FOR INFORMATION: WE CHECK IF WE MATCH THE EUROSTAT NAMES
#-------------------------------------------------------------------------------
    if check_eurostat_file == True:
        check_EUROSTAT_names_match(NUTS_names_dict, crop_names_dict, EUROSTATdir)


#===============================================================================
def map_crop_id_to_crop_name(crop_list):
#===============================================================================
    # dict[crop_short_name] = [CGMS_id_nb, EUROSTAT_name1, EUROSTAT_name2]
    crop_names = dict()

    for nickname in crop_list:
        if (nickname == 'Winter wheat'):
            crop_names[nickname] = [1,'Common wheat and spelt',
                                      'Common winter wheat']
        if (nickname == 'Spring wheat'):
            crop_names[nickname] = [np.nan,'Common wheat and spelt',
                                      'Common spring wheat']
        #if (nickname == 'durum wheat'):
        #    crop_names[nickname] = [np.nan,'Durum wheat',
        #                               'Durum wheat']
        if (nickname == 'Grain maize'):
            crop_names[nickname] = [2,'Grain maize',
                                      'Grain maize and corn-cob-mix']
        if (nickname == 'Fodder maize'):
            crop_names[nickname] = [12,'Green maize',
                                      'Green maize']
        if (nickname == 'Spring barley'):
            crop_names[nickname] = [3,'Barley',
                                      'Barley']
        if (nickname == 'Winter barley'):
            crop_names[nickname] = [13,'Barley',
                                      'Winter barley']
        if (nickname == 'Rye'):
            crop_names[nickname] = [4,'Rye',
                                      'Rye']
        #if (nickname == 'Rice'):
        #    crop_names[nickname] = [np.nan,'Rice',
        #                              'Rice']
        if (nickname == 'Sugar beet'):
            crop_names[nickname] = [6,'Sugar beet (excluding seed)',
                                      'Sugar beet (excluding seed)']
        if (nickname == 'Potato'):
            crop_names[nickname] = [7,'Potatoes (including early potatoes and'+\
                                      ' seed potatoes)','Potatoes (including '+\
                                      'early potatoes and seed potatoes)']
        if (nickname == 'Field beans'):
            crop_names[nickname] = [8,'Dried pulses and protein crops for the'+\
                                      ' production of grain (including seed ' +\
                                      'and mixtures of cereals and pulses)',
                                      'Broad and field beans']
        if (nickname == 'Spring rapeseed'):
            crop_names[nickname] = [np.nan,'Rape and turnip rape',
                                      'Spring rape']
        if (nickname == 'Winter rapeseed'):
            crop_names[nickname] = [10,'Rape and turnip rape',
                                      'Winter rape']
        if (nickname == 'Sunflower'):
            crop_names[nickname] = [11,'Sunflower seed',
                                      'Sunflower seed']

    print '\nWe select the following crops:\n', sorted(crop_names.keys())

    return crop_names

#===============================================================================
def map_NUTS_id_to_NUTS_name(list_of_NUTS_ids, EUROSTATdir):
#===============================================================================
    from maries_toolbox import fetch_EUROSTAT_NUTS_name

    dict_geo_units = dict()

    for NUTS_no in list_of_NUTS_ids:
        try:
            NUTS_name_dict = fetch_EUROSTAT_NUTS_name(NUTS_no, EUROSTATdir)
            NUTS_name = NUTS_name_dict[NUTS_no]
			# we correct the alphabet of the following region names in the
			# dictionary:
            if (NUTS_no == 'BG31'): NUTS_name = "Severozapaden"
            if (NUTS_no == 'BG32'): NUTS_name = "Severen tsentralen"
            if (NUTS_no == 'BG33'): NUTS_name = "Severoiztochen"
            if (NUTS_no == 'BG34'): NUTS_name = "Yugoiztochen"
            if (NUTS_no == 'BG41'): NUTS_name = "Yugozapaden"
            if (NUTS_no == 'BG42'): NUTS_name = "Yuzhen tsentralen"
            if (NUTS_no == 'EL30'): NUTS_name = "Attiki"
            if (NUTS_no == 'EL41'): NUTS_name = "Voreio Aigaio"
            if (NUTS_no == 'EL42'): NUTS_name = "Notio Aigaio"
            if (NUTS_no == 'EL43'): NUTS_name = "Kriti"
            if (NUTS_no == 'PL21'): NUTS_name = "Malopolskie"
            if (NUTS_no == 'PL22'): NUTS_name = "Slaskie"
            if (NUTS_no == 'PL33'): NUTS_name = "Swietokrzyskie"
            if (NUTS_no == 'PL51'): NUTS_name = "Dolnoslaskie"
            if (NUTS_no == 'PL62'): NUTS_name = "Warminsko-mazurskie"
            if (NUTS_no == 'RO32'): NUTS_name = "Bucuresti - Ilfov"
            # we select the english name:
            if (NUTS_no == 'CY00'): NUTS_name = "CYPRUS" 
                                    # 'Cyprus' or 'Kypros' in EUROSTAT
            if (NUTS_no == 'CZ0'): NUTS_name = "CZECH REPUBLIC" 
                                    # 'Czech Republic' or 'Cesk\xe1 republika'
            # for all previous examples, EURO_name = NUTS_name:
            EURO_name = NUTS_name.lower()
			# we correct the accents of the following region names in the
			# dictionary:
            if (NUTS_no == 'AT1'): 
                NUTS_name = "Ostosterreich"
                EURO_name = "Ost\xf6sterreich"
            if (NUTS_no == 'AT2'): 
                NUTS_name = "Sudosterreich"
                EURO_name = "S\xfcd\xf6sterreich"
            if (NUTS_no == 'AT3'): 
                NUTS_name = "Westosterreich"
                EURO_name = "West\xf6sterreich"
            if (NUTS_no == 'BE1'): 
                NUTS_name = "Region de Bruxelles-capitale / Brussels "\
                           +"Hoofdstedelijk gewest"
                EURO_name = "R\xe9gion de Bruxelles-Capitale / Brussels "\
                           +"Hoofdstedelijk Gewest"
            if (NUTS_no == 'BE3'): 
                NUTS_name = "Region Wallonne"
                EURO_name = "R\xe9gion wallonne"
            if (NUTS_no == 'DE1'): 
                NUTS_name = "Baden-Wurttemberg"
                EURO_name = "Baden-W\xfcrttemberg"
            if (NUTS_no == 'DEG'): 
                NUTS_name = "Thuringen"
                EURO_name = "Th\xfcringen"
            if (NUTS_no == 'DK02'): 
                NUTS_name = "Sjaelland"
                EURO_name = "Sj\xe6lland"
            if (NUTS_no == 'ES21'): 
                NUTS_name = "Pais Vasco"
                EURO_name = "Pa\xeds Vasco"
            if (NUTS_no == 'ES24'): 
                NUTS_name = "Aragon"
                EURO_name = "Arag\xf3n"
            if (NUTS_no == 'ES41'): 
                NUTS_name = "Castilla y Leon"
                EURO_name = "Castilla y Le\xf3n"
            if (NUTS_no == 'ES51'): 
                NUTS_name = "Cataluna"
                EURO_name = "Catalu\xf1a"
            if (NUTS_no == 'ES61'): 
                NUTS_name = "Andalucia"
                EURO_name = "Andaluc\xeda"
            if (NUTS_no == 'ES62'): 
                NUTS_name = "Region de Murcia"
                EURO_name = "Regi\xf3n de Murcia"
            if (NUTS_no == 'ES63'): 
                NUTS_name = "Ciudad Autonoma de Ceuta"
                EURO_name = "Ciudad Aut\xf3noma de Ceuta (ES)"
            if (NUTS_no == 'ES64'): 
                NUTS_name = "Ciudad Autonoma de Melilla"
                EURO_name = "Ciudad Aut\xf3noma de Melilla (ES)"
            if (NUTS_no == 'FI19'): 
                NUTS_name = "Lansi-Suomi"
                EURO_name = "L\xe4nsi-Suomi"
            if (NUTS_no == 'FI1C'): 
                NUTS_name = "Etela-Suomi"
                EURO_name = "Etel\xe4-Suomi"
            if (NUTS_no == 'FI1D'): 
                NUTS_name = "Pohjois- ja Ita-Suomi"
                EURO_name = "Pohjois- ja It\xe4-Suomi"#"It\xe4-Suomi (NUTS 2006)"
            if (NUTS_no == 'FI20'): 
                NUTS_name = "Aland"
                EURO_name = "\xc5land"
            if (NUTS_no == 'FR10'): 
                NUTS_name = "Ile de France"
                EURO_name = "\xcele de France"
            if (NUTS_no == 'FR43'): 
                NUTS_name = "Franche-Comte"
                EURO_name = "Franche-Comt\xe9"
            if (NUTS_no == 'FR62'): 
                NUTS_name = "Midi-Pyrenees"
                EURO_name = "Midi-Pyr\xe9n\xe9es"
            if (NUTS_no == 'FR71'): 
                NUTS_name = "Rhone-Alpes"
                EURO_name = "Rh\xf4ne-Alpes"
            if (NUTS_no == 'FR82'): 
                NUTS_name = "Provence-Alpes-Cote d'Azur"
                EURO_name = "Provence-Alpes-C\xf4te d'Azur"
            if (NUTS_no == 'HU10'): 
                NUTS_name = "Kozep-Magyarorszag"
                EURO_name = "K\xf6z\xe9p-Magyarorsz\xe1g"
            if (NUTS_no == 'HU21'): 
                NUTS_name = "Kozep-Dunantul"
                EURO_name = "K\xf6z\xe9p-Dun\xe1nt\xfal"
            if (NUTS_no == 'HU22'): 
                NUTS_name = "Nyugat-Dunantul"
                EURO_name = "Nyugat-Dun\xe1nt\xfal"
            if (NUTS_no == 'HU23'): 
                NUTS_name = "Del-Dunantul"
                EURO_name = "D\xe9l-Dun\xe1nt\xfal"
            if (NUTS_no == 'HU31'): 
                NUTS_name = "Eszak-Magyarorszag"
                EURO_name = "\xc9szak-Magyarorsz\xe1g"
            if (NUTS_no == 'HU32'): 
                NUTS_name = "Eszak-Alfold"
                EURO_name = "\xc9szak-Alf\xf6ld"
            if (NUTS_no == 'HU33'): 
                NUTS_name = "Del-Alfold"
                EURO_name = "D\xe9l-Alf\xf6ld"
            if (NUTS_no == 'ITC2'): 
                NUTS_name = "Valle d'Aosta/Vallee d'Aoste"
                EURO_name = "Valle d'Aosta/Vall\xe9e d'Aoste"
            if (NUTS_no == 'PL11'): 
                NUTS_name = "Lodzkie"
                EURO_name = "L\xf3dzkie"
            if (NUTS_no == 'PT17'): 
                NUTS_name = "Area Metropolitana de Lisboa"
                EURO_name = "\xc1rea Metropolitana de Lisboa"
            if (NUTS_no == 'PT20'): 
                NUTS_name = "Regiao Autonoma dos Acores"
                EURO_name = "Regi\xe3o Aut\xf3noma dos A\xe7ores (PT)"
            if (NUTS_no == 'PT30'): 
                NUTS_name = "Regiao Autonoma da Madeira"
                EURO_name = "Regi\xe3o Aut\xf3noma da Madeira (PT)"
            if (NUTS_no == 'SE12'): 
                NUTS_name = "Ostra Mellansverige"
                EURO_name = "\xd6stra Mellansverige"
            if (NUTS_no == 'SE21'): 
                NUTS_name = "Smaland med oarna"
                EURO_name = "Sm\xe5land med \xf6arna"
            if (NUTS_no == 'SE23'): 
                NUTS_name = "Vastsverige"
                EURO_name = "V\xe4stsverige"
            if (NUTS_no == 'SE33'): 
                NUTS_name = "Ovre Norrland"
                EURO_name = "\xd6vre Norrland"
            if (NUTS_no == 'SK01'): 
                NUTS_name = "Bratislavsky kraj"
                EURO_name = "Bratislavsk\xfd kraj"
            if (NUTS_no == 'SK02'): 
                NUTS_name = "Zapadne Slovensko"
                EURO_name = "Z\xe1padn\xe9 Slovensko"
            if (NUTS_no == 'SK03'): 
                NUTS_name = "Stredne Slovensko"
                EURO_name = "Stredn\xe9 Slovensko"
            if (NUTS_no == 'SK04'): 
                NUTS_name = "Vychodne Slovensko"
                EURO_name = "V\xfdchodn\xe9 Slovensko"

            # we correct exceptions: same regions called with different name in
            # EUROSTAT records:
            if (NUTS_no == 'ES70'): EURO_name = "Canarias (ES)"
            if (NUTS_no == 'FR24'): EURO_name = "Centre (FR)"
            if (NUTS_no == 'ITH5'): EURO_name = "Emilia-Romagna (NUTS 2006)"
            if (NUTS_no == 'ITI3'): EURO_name = "Marche (NUTS 2006)"
            if (NUTS_no == 'UKC'): EURO_name = "North East (UK)"
            if (NUTS_no == 'UKD'): EURO_name = "North West (UK)"
            if (NUTS_no == 'UKF'): EURO_name = "East Midlands (UK)"
            if (NUTS_no == 'UKG'): EURO_name = "West Midlands (UK)"
            if (NUTS_no == 'UKJ'): EURO_name = "South East (UK)"
            if (NUTS_no == 'UKK'): EURO_name = "South West (UK)"
            if (NUTS_no == 'UKN'): EURO_name = "Northern Ireland (UK)"
            if (NUTS_no == 'HR03'): EURO_name = "Jadranska Hrvatska"
            if (NUTS_no == 'HR04'): EURO_name = "Beware: HR04 = HR01 + HR02!!"
        # if the region is not found in the NUTS codes csv file, we add the region
        # by hand:
        except KeyError:
            # for every NUTS id without a name we write 'unknown'
            NUTS_name = 'unknown'
            if (NUTS_no == 'CH0'): NUTS_name = "SWITZERLAND"
            if (NUTS_no == 'LI0'): NUTS_name = "LIECHTENSTEIN"
            if (NUTS_no == 'ME0'): NUTS_name = "MONTENEGRO"
            if (NUTS_no == 'EL11'): NUTS_name = "Anatoliki Makedonia, Thraki" # new: EL51
            if (NUTS_no == 'EL12'): NUTS_name = "Kentriki Makedonia" #new: EL52
            if (NUTS_no == 'EL13'): NUTS_name = "Dytiki Makedonia" # new: EL53
            if (NUTS_no == 'EL14'): NUTS_name = "Thessalia"
            if (NUTS_no == 'EL21'): NUTS_name = "Ipeiros"
            if (NUTS_no == 'EL22'): NUTS_name = "Ionia Nisia"
            if (NUTS_no == 'EL23'): NUTS_name = "Dytiki Ellada"
            if (NUTS_no == 'EL24'): NUTS_name = "Sterea Ellada"
            if (NUTS_no == 'EL25'): NUTS_name = "Peloponnisos"
            if (NUTS_no == 'FR91'): NUTS_name = "Guadeloupe"
            if (NUTS_no == 'FR92'): NUTS_name = "Martinique"
            if (NUTS_no == 'FR93'): NUTS_name = "Guyane"
            if (NUTS_no == 'FR94'): NUTS_name = "Reunion"
            if (NUTS_no == 'IS00'): NUTS_name = "ICELAND"
            if (NUTS_no == 'MK00'): NUTS_name = "MACEDONIA"
            if (NUTS_no == 'NO01'): NUTS_name = "Oslo og Akershus"
            if (NUTS_no == 'NO02'): NUTS_name = "Hedmark og Oppland"
            if (NUTS_no == 'NO03'): NUTS_name = "Sor-Ostlandet"
            if (NUTS_no == 'NO04'): NUTS_name = "Agder og Rogaland"
            if (NUTS_no == 'NO05'): NUTS_name = "Vestlandet"
            if (NUTS_no == 'NO06'): NUTS_name = "Trondelag"
            if (NUTS_no == 'NO07'): NUTS_name = "Nord-Norge"
            if (NUTS_no == 'SI01'): NUTS_name = "Vzhodna Slovenija"
            if (NUTS_no == 'SI02'): NUTS_name = "Zahodna Slovenija"
            if (NUTS_no == 'TR10'): NUTS_name = "Istanbul"
            if (NUTS_no == 'TR21'): NUTS_name = "Tekirdag"
            if (NUTS_no == 'TR22'): NUTS_name = "Balikesir"
            if (NUTS_no == 'TR31'): NUTS_name = "Izmir"
            if (NUTS_no == 'TR32'): NUTS_name = "Aydin"
            if (NUTS_no == 'TR33'): NUTS_name = "Manisa"
            if (NUTS_no == 'TR41'): NUTS_name = "Bursa"
            if (NUTS_no == 'TR42'): NUTS_name = "Kocaeli"
            if (NUTS_no == 'TR51'): NUTS_name = "Ankara"
            if (NUTS_no == 'TR52'): NUTS_name = "Konya"
            if (NUTS_no == 'TR61'): NUTS_name = "Antalya"
            if (NUTS_no == 'TR62'): NUTS_name = "Adana"
            if (NUTS_no == 'TR63'): NUTS_name = "Hatay"
            if (NUTS_no == 'TR71'): NUTS_name = "Kirikkale"
            if (NUTS_no == 'TR72'): NUTS_name = "Kayseri"
            if (NUTS_no == 'TR81'): NUTS_name = "Zonguldak"
            if (NUTS_no == 'TR82'): NUTS_name = "Kastamonu"
            if (NUTS_no == 'TR83'): NUTS_name = "Samsun"
            if (NUTS_no == 'TR90'): NUTS_name = "Trabzon"
            if (NUTS_no == 'TRA1'): NUTS_name = "Erzurum"
            if (NUTS_no == 'TRA2'): NUTS_name = "Agri"
            if (NUTS_no == 'TRB1'): NUTS_name = "Malatya"
            if (NUTS_no == 'TRB2'): NUTS_name = "Van"
            if (NUTS_no == 'TRC1'): NUTS_name = "Gaziantep"
            if (NUTS_no == 'TRC2'): NUTS_name = "Sanliurfa"
            if (NUTS_no == 'TRC3'): NUTS_name = "Mardin"
            # for all previous examples, EURO_name = NUTS_name:
            EURO_name = NUTS_name.lower()
            # we correct the accents of the following region names in the dictionary:
            if (NUTS_no == 'NO03'): EURO_name = "S\xf8r-\xd8stlandet"
            if (NUTS_no == 'NO06'): EURO_name = "Tr\xf8ndelag"
            # we correct exceptions: same regions called with different name in
            # EUROSTAT records:
            if (NUTS_no == 'EL11'): EURO_name = "Anatoliki Makedonia, Thraki (NUTS 2010)" # new: EL51
            if (NUTS_no == 'EL12'): EURO_name = "Kentriki Makedonia (NUTS 2010)" # new: EL52
            if (NUTS_no == 'EL13'): EURO_name = "Dytiki Makedonia (NUTS 2010)" # new: EL53
            if (NUTS_no == 'EL14'): EURO_name = "Thessalia (NUTS 2010)"
            if (NUTS_no == 'EL21'): EURO_name = "Ipeiros (NUTS 2010)"
            if (NUTS_no == 'EL22'): EURO_name = "Ionia Nisia (NUTS 2010)"
            if (NUTS_no == 'EL23'): EURO_name = "Dytiki Ellada (NUTS 2010)"
            if (NUTS_no == 'EL24'): EURO_name = "Sterea Ellada (NUTS 2010)"
            if (NUTS_no == 'EL25'): EURO_name = "Peloponnisos (NUTS 2010)"
            if (NUTS_no == 'FR91'): EURO_name = "Guadeloupe (NUTS 2010)"
            if (NUTS_no == 'FR92'): EURO_name = "Martinique (NUTS 2010)"
            if (NUTS_no == 'FR93'): EURO_name = "Guyane (NUTS 2010)"
            if (NUTS_no == 'FR94'): EURO_name = "R\xe9union (NUTS 2010)"
            if (NUTS_no == 'MK00'): EURO_name = "Former Yugoslav Republic of Macedonia, the"
            if (NUTS_no == 'SI01'): EURO_name = "Vzhodna Slovenija (NUTS 2010)"
            if (NUTS_no == 'SI02'): EURO_name = "Zahodna Slovenija (NUTS 2010)"
            if (NUTS_no == 'TR21'): EURO_name = "Tekirdag, Edirne, Kirklareli"
            if (NUTS_no == 'TR22'): EURO_name = "Balikesir, \xc7anakkale"
            if (NUTS_no == 'TR32'): EURO_name = "Aydin, Denizli, Mugla"
            if (NUTS_no == 'TR33'): EURO_name = "Manisa, Afyonkarahisar, K\xfctahya, Usak"
            if (NUTS_no == 'TR41'): EURO_name = "Bursa, Eskisehir, Bilecik"
            if (NUTS_no == 'TR42'): EURO_name = "Kocaeli, Sakarya, D\xfczce, Bolu, Yalova"
            if (NUTS_no == 'TR52'): EURO_name = "Konya, Karaman"
            if (NUTS_no == 'TR61'): EURO_name = "Antalya, Isparta, Burdur"
            if (NUTS_no == 'TR62'): EURO_name = "Adana, Mersin"
            if (NUTS_no == 'TR63'): EURO_name = "Hatay, Kahramanmaras, Osmaniye"
            if (NUTS_no == 'TR71'): EURO_name = "Kirikkale, Aksaray, Nigde, Nevsehir, Kirsehir"
            if (NUTS_no == 'TR72'): EURO_name = "Kayseri, Sivas, Yozgat"
            if (NUTS_no == 'TR81'): EURO_name = "Zonguldak, Karab\xfck, Bartin"
            if (NUTS_no == 'TR82'): EURO_name = "Kastamonu, \xc7ankiri, Sinop"
            if (NUTS_no == 'TR83'): EURO_name = "Samsun, Tokat, \xc7orum, Amasya"
            if (NUTS_no == 'TR90'): EURO_name = "Trabzon, Ordu, Giresun, Rize, Artvin, G\xfcm\xfcshane"
            if (NUTS_no == 'TRA1'): EURO_name = "Erzurum, Erzincan, Bayburt"
            if (NUTS_no == 'TRA2'): EURO_name = "Agri, Kars, Igdir, Ardahan"
            if (NUTS_no == 'TRB1'): EURO_name = "Malatya, Elazig, Bing\xf6l, Tunceli"
            if (NUTS_no == 'TRB2'): EURO_name = "Van, Mus, Bitlis, Hakkari"
            if (NUTS_no == 'TRC1'): EURO_name = "Gaziantep, Adiyaman, Kilis"
            if (NUTS_no == 'TRC2'): EURO_name = "Sanliurfa, Diyarbakir"
            if (NUTS_no == 'TRC3'): EURO_name = "Mardin, Batman, Sirnak, Siirt"
            # no EUROSTAT record:
            if (NUTS_no == 'CH0'): EURO_name = "Schweiss / Suisse / Svizzera"
        entry = [NUTS_name,EURO_name]
        dict_geo_units[NUTS_no]=entry
    # we remove some entries always absent in EUROSTAT records:
    del dict_geo_units['CH0']
    del dict_geo_units['LI0']
    del dict_geo_units['HR04']
    # we add some necessary entries:
    dict_geo_units['HR01']=["Sjeverozapadna Hrvatska",
                            "Sjeverozapadna Hrvatska (former "\
                                               +"statistical region)"]
    dict_geo_units['HR02']=["Sredisnja i Istocna (Panonska) Hrvatska",
                            "Sredisnja i Istocna (Panonska) "\
                                               +"Hrvatska (former statistical "\
                                               +"region)"]

    print '\nWe select the following NUTS regions:\n', sorted(dict_geo_units.keys())

    return dict_geo_units

#===============================================================================
def make_NUTS_composite(lands_levels, EUROSTATdir):
#===============================================================================

    from mpl_toolkits.basemap import Basemap

    map = Basemap(projection='laea', lat_0=48, lon_0=16, llcrnrlat=30, 
                      llcrnrlon=-10, urcrnrlat=65, urcrnrlon=45)
    # Read a shapefile and its metadata
    # read the shapefile data WITHOUT plotting its shapes

    path = EUROSTATdir
    filename = 'NUTS_RG_03M_2010'# NUTS regions 
    name = 'NUTS'
    NUTS_info = map.readshapefile(path + filename, name, drawbounds=False) 

    # retrieve the list of patches to fill and its data to plot
    NUTS_ids_list = list()
    # for each polygon of the shapefile
    for info, shape in zip(map.NUTS_info, map.NUTS):
        # we get the NUTS number of this polygon:
        NUTS_no = info['NUTS_ID']
        # if the NUTS level of the polygon corresponds to the desired one:
        if (info['STAT_LEVL_'] == lands_levels[NUTS_no[0:2]]):
            if NUTS_no not in NUTS_ids_list:
                NUTS_ids_list += [NUTS_no]

    return NUTS_ids_list

#===============================================================================
def check_EUROSTAT_names_match(NUTS_names_dict, crop_names_dict, EUROSTATdir):
#===============================================================================
    from maries_toolbox import open_csv_EUROSTAT

	# read the yield/ landuse / pop records, try to match previous name with
	# names in this file
    NUTS_filenames = ['agri_yields_NUTS1-2-3_1975-2014.csv',
                     'agri_croparea_NUTS1-2-3_1975-2014.csv',
                     'agri_landuse_NUTS1-2-3_2000-2013.csv',
                     'agri_prodhumid_NUTS1_1955-2015.csv',
                     'agri_harvest_NUTS1-2-3_1975-2014.csv']
    NUTS_data = open_csv_EUROSTAT(EUROSTATdir, NUTS_filenames,
                                     convert_to_float=True, verbose=False)

    # read the NUTS region labels for each EUROSTAT csv file:
    geo_dict = dict()
    crop_dict = dict()
    for record in NUTS_filenames:
        if (record != 'agri_prodhumid_NUTS1_1955-2015.csv'):
            geo_units = NUTS_data[record]['GEO']
            geo_units = list(set(geo_units))
            geo_dict[record] = [u.lower() for u in geo_units]
            #print geo_units, len (geo_units)
        if (record != 'agri_landuse_NUTS1-2-3_2000-2013.csv'):
            crop_names = NUTS_data[record]['CROP_PRO']
            crop_names = list(set(crop_names))
            crop_dict[record] = [u.lower() for u in crop_names]
            #print record, crop_names, len (crop_names)

	# Check if we match all EUROSTAT region names, in all the EUROSTAT
	# observation files:
    for record in NUTS_filenames:
        print '\nChecking EUROSTAT file %s'%record
        if (record != 'agri_prodhumid_NUTS1_1955-2015.csv'):
            counter = 0
            for key in NUTS_names_dict.keys():
                if (NUTS_names_dict[key][1].lower() not in geo_dict[record]):
                    counter +=1
                    if counter ==1: print '        NUTS ID,   Corrected name,   EUROSTAT name'
                    print 'NOT OK: %5s'%key, NUTS_names_dict[key]
            print 'found %i unmatched region names in EUROSTAT file'%counter
        if (record != 'agri_landuse_NUTS1-2-3_2000-2013.csv'):
            if (record == 'agri_prodhumid_NUTS1_1955-2015.csv'):
                counter = 0
                for key in crop_names_dict.keys():
                    if (crop_names_dict[key][2].lower() not in crop_dict[record]):
                        counter +=1
                        if counter ==1: print '        crop ID,   Corrected name,   EUROSTAT name'
                        print 'NOT OK: %5s'%key, crop_names_dict[key]
                print 'found %i unmatched crop names in EUROSTAT file'%counter
            else:
                counter = 0
                for key in crop_names_dict.keys():
                    if (crop_names_dict[key][1].lower() not in crop_dict[record]):
                        counter +=1
                        if counter ==1: print '        crop ID,   Corrected name,   EUROSTAT name'
                        print 'NOT OK: %5s'%key, crop_names_dict[key]
                print 'found %i unmatched crop names in EUROSTAT file'%counter

    return None

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================

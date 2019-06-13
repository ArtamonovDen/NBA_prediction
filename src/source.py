
import pandas as pd
import numpy as np
import time
import pickle
import time
from datetime import datetime
import  multiprocessing as mlp

from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints.playercareerstats import PlayerCareerStats
from nba_api.stats.endpoints.commonteamroster import  CommonTeamRoster
from nba_api.stats.endpoints.playergamelog import  PlayerGameLog


from sklearn import linear_model
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, r2_score, mean_squared_error

import lightgbm as lgb

import seaborn as sns
import matplotlib.pyplot as plt


#functions for fast model scoring


def score_model(y_pred, y_true):
    average = 'weighted'
    f1 = f1_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    return f1, precision, recall

def check_model(y_pred,Y_test):
    f1, precision, recall = score_model(y_pred,Y_test)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'f1-score: {f1:.4f}')





def mlt_players_games(player):
    '''
        Returns pd.DataFrame with all player's games per season

        Change SEASONE global var!
    '''
    season = SEASON
    time.sleep(datetime.now().microsecond & 7)
    print('     ' + 'load games for ' + str(player))
    time.sleep(datetime.now().microsecond & 3)
    games = PlayerGameLog(player_id = player, season_all = season).get_data_frames()[0]
    time.sleep(datetime.now().microsecond & 7)
    return games



def mlt_team_roster(team):
    '''
        Returns pd.DataFrame with [TeamID, PlayerID] columns

        Change SEASONE global var!
    '''
    season = SEASON
    #print(season)
    
    time.sleep(datetime.now().microsecond & 7)
    col = ['TeamID', 'PLAYER_ID']
    print('     ' + 'load roster for ' + str(team))
    time.sleep(datetime.now().microsecond & 3)
    roster = CommonTeamRoster(team_id = team, season = season).get_data_frames()[0][col]
    time.sleep(datetime.now().microsecond & 7)
    return roster



def download_in_parallel(apply_to_list, f):
    '''
        Apply specified function f to list apply_to_list.
        Returns joint pd.DataFrame
    '''    

    pool = mlp.Pool(processes = 3)
    outputs = pool.map(f, apply_to_list)
    return pd.concat(outputs, ignore_index=True)



def download_team_rosters(teams, season = '2018-19', save = True):
    '''
        Return list of rosters for specified teams
    '''
    global SEASON
    SEASON = season


    rosters = download_in_parallel(teams, mlt_team_roster)
    result = list()
    if save:
        for team in teams:
            roster = rosters[rosters['TeamID']==team]
            result.append(roster)
            serialize(roster, './data/rosters/', season + '_roster_' + str(team))
    
    

def download_players_stat(teams, season):
    '''
        For players in teams' rosters it downloads all games per season
    '''    
    global SEASON
    SEASON = season
    rosters_dir = root_path + 'data/rosters/'
    games_dir = root_path + 'data/player_games/'
    for team in teams:
        print('load for team ' + str(team))
        #filename = 'roster_' + str(team) + ".pickle"
        filename = season + '_roster_' + str(team) + ".pickle"
        roster = pd.read_pickle(rosters_dir + filename)['PLAYER_ID']
        all_games = download_in_parallel(roster, mlt_players_games)
        serialize(all_games, games_dir, season + '_allgames_' + str(team))
   


def preproc_build_team_games_dict(teams,season = '2018-19', save = True):
    '''
        Creates map
        {
            team : all_games
        }
        where all_games is joint pd.DataFrame of all players' games per season
    '''
    team_games_dict = dict()
    games_dir = root_path + 'data/player_games/'
    for team in teams:
        filename = season + '_allgames_' + str(team) + ".pickle"
        all_games = pd.read_pickle(games_dir + filename )
        team_games_dict[team] = all_games
    if save:
        with open (root_path + "temp/"+ season +"_team_games_dict" + ".pickle", 'wb+') as f:
            pickle.dump(team_games_dict, f)

    return team_games_dict



def preproc_build_player_games(data:pd.DataFrame, team_games_dict,  season = '2018-19', save = True):
    
    '''
        Creates map
        {
            game_id: {
                one_team_id{
                    player_id : stat_per_game
                    ...
                }
                two_team_id{
                    player_id : stat_per_game
                    ...
                }             
            }
            ...
        }

    '''    
    player_stat_columns = ['Player_ID', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS']

    big_players_games_stat = dict()
    
    for game in data.iterrows():

        game_id = game[1]['Game_ID']
        team_id = game[1]['Team_ID']

        all_games = team_games_dict[team_id]
        players_stat = all_games[all_games['Game_ID'] == game_id]
        players_stat = players_stat[player_stat_columns]
        game_team_dict = dict()
        for stat in players_stat.iterrows():
            player_id = int(stat[1]['Player_ID'])
            game_team_dict[player_id] = stat[1].drop(['Player_ID'])

        if game_id in big_players_games_stat:
            big_players_games_stat[game_id][team_id] = game_team_dict
        else:
            big_players_games_stat[game_id] = {team_id:game_team_dict}

    if save:
        with open (root_path + "temp/big_players_games_stat_" + season + ".pickle", 'wb+') as f:
            pickle.dump(big_players_games_stat, f)
    return big_players_games_stat


def to_train_crew(data:pd.DataFrame, big_players_games_stat, fts=None):
    '''
        Maps each game with its stat, obtained as residual between mean stat of players of both teams.
        Also there is column TRG with {0,1} detecting win or loss

        Args:
            data - [game_id - team_id - 'PTS']
    '''

    if fts is None:
        features = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
        'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
        'PTS', 'TRG']
    else:
        features = fts



    data = data[['Team_ID', 'Game_ID', 'PTS' ]]  
    data = data.set_index( ['Game_ID', 'Team_ID'], drop = True ) 

    train_data = pd.DataFrame(columns = features)

    for game, teams_games in big_players_games_stat.items():
        mean_stats = list()
        team_pts = list()
        for team, players_stat in teams_games.items():
            mean_stats.append(pd.DataFrame(players_stat.values()).mean())
            team_pts.append(data.loc[game].loc[team])

        game_stat = mean_stats[0] - mean_stats[1]
        trg = np.sign(team_pts[0] - team_pts[1]).rename( {'PTS' : 'TRG'}) # MAY BE CHANGED TO REGR
        train_data = train_data.append(game_stat.append(trg), ignore_index = True)

    X_train = choose_features(features, train_data).values.astype('float64')
    return X_train[:,:-1], X_train[:,-1].astype('int')


def preproc_build_df_crew(data:pd.DataFrame, big_players_games_stat, season, save = True):
    '''
        Build pd.DataFrame :
        game - team - mean stata by players 
        
    '''
    columns = ['Game_ID', 'Team_ID','FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
       'PTS', 'TRG']

    data = data[['Team_ID', 'Game_ID', 'PTS' ]]  
    data = data.set_index( ['Game_ID', 'Team_ID'], drop = True ) 

    crew_data = pd.DataFrame(columns = columns)

    for game, teams_games in big_players_games_stat.items():
        for team, players_stat in teams_games.items():
            mean_stats = pd.DataFrame(players_stat.values()).mean()
            trg = data.loc[game].loc[team].rename( {'PTS' : 'TRG'})
            meta = pd.Series({'Game_ID':game, 'Team_ID':team})
            crew_data = crew_data.append( mean_stats.append([meta, trg]), ignore_index = True)
        
    if save:
        serialize(crew_data, dir_path = root_path + "data/crew_data/", filename = "crew_data_"+season)
    return crew_data


def preproc_mean_stat_crew(season = '2017-18' ):
    '''
        Build pd.DataFrame:
        team_id -> mean team by players
    '''
    filename = root_path + "/data/all_games_per_seasone/mean_players_stat_per" + season + ".pickle"
    middle_stat_per_players = pd.read_pickle(filename)


    stat_columns = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
       'PTS']
    columns = ['Team_ID', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
       'PTS']
    middle_stat_per_players = middle_stat_per_players.set_index(['TeamID', 'PLAYER_ID'], drop=False)
    teams = list(set(middle_stat_per_players['TeamID']))
    mean_stat_crew = pd.DataFrame(columns=columns)
    for team in teams:
        mean_stat = middle_stat_per_players.loc[team][stat_columns].mean()
        mean_stat_crew = mean_stat_crew.append(
            pd.Series({'Team_ID' : team}).append([mean_stat]), ignore_index=True
            )

               
    mean_stat_crew['Team_ID'] = mean_stat_crew['Team_ID'].astype('int64') 
    mean_stat_crew = mean_stat_crew.set_index(['Team_ID'])
    return mean_stat_crew

def button_test_crew(season, last_season, features = None):
    crew_df = pd.read_pickle(root_path + 'data/crew_data/crew_data_' + season + '.pickle')
    with open(root_path + 'temp/'+season+'_crew_dict.pickle' , 'rb') as f:
        crew_dict = pickle.load(f)
    return to_test_crew(crew_df, features, crew_dict, season,last_season)

def to_test_crew(crew_data:pd.DataFrame, features = None, crew_dict = None, test_season = '2018-19', last_season = '2017-18'):
    '''
        Build tets sample as mean( mean_of_last_season(by team),  games before in season)

        Args:
            crew_data -result of preproc_build_df_crew()
    '''
   


    if crew_dict is None:
        with open(root_path+"temp/crew_dict.pickle", 'rb') as f:
            crew_dict = pickle.load(f)

    test_data_crew= pd.DataFrame(columns=crew_data.columns)

    #test_data_crew - replace actual stat for game with mean stat
    for game in crew_data.iterrows():
        team_id = game[1]['Team_ID']
        game_id = game[1]['Game_ID']
        mean_game = crew_dict[team_id][game_id]
        mean_game['TRG'] = game[1]['TRG'] # mean prepared stats but actual target
        test_data_crew = test_data_crew.append(mean_game, ignore_index=True)
 
    test_data_crew = prepare_features(test_data_crew, drop_col = True, col_to_drop = ['Team_ID', 'Game_ID']) 

    if features is None:
        features = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
        'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
        'PTS', 'TRG']

    X_test_window = choose_features(features, test_data_crew).values.astype('float64')
    #X_test_window = test_data_crew.values.astype('float64')

    return X_test_window[:,:-1], X_test_window[:,-1].astype('int')

def build_crew_dict(data:pd.DataFrame, season = None, save = True):
    '''
        Maps every game of each command with features it is described with.
        Creates dict:
        { 
            team_id :{
                game_id : mean stat by players of last games in season and mean stat per team in last season
            }           
        }
        Note, that stat has only chosen features
        Data is concat mean_stat_crew and crew_data
        
        
    '''

    # mean_stat_crew = preproc_mean_stat_crew(last_season).reset_index()
    # data = mean_stat_crew.append(crew_data, sort=False)

    f_to_mean = [ 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
       'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
       'PF', 'PTS', 'TRG']
    crew_dict = {}
    teams = set(data['Team_ID'])
    for team in teams:
            crew_dict[team] = get_dict_moving_ave_per_team(data, team, 'TRG', f_to_mean )

    if save:
        with open (root_path+'temp/' + season + '_crew_dict.pickle', 'wb+') as f:
            pickle.dump(crew_dict, f)

    return crew_dict 

    

def preproc_players_season_stat( teams, season = '2017-18'):

    '''
        For every team in teams it obtains list of players. Then for 
        every player it gets his mean stats per specified season( stats / num of games).

        Returns pd.DataFrame comtaining season mean stat for every player in teams

    '''
    joined_columns = ['TeamID', 'SEASON', 'LeagueID', 'PLAYER', 'NUM', 'POSITION', 'HEIGHT',
       'WEIGHT', 'BIRTH_DATE', 'AGE', 'EXP', 'SCHOOL', 'PLAYER_ID',
       'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION',
       'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS']
    player_stat_columns = ['PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'TEAM_ABBREVIATION',
       'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS']
    to_mean_features = ['FGM', 'FGA', 'FG3M', 'FG3A','FTM', 'FTA',  'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', 'PTS']

    

    joined_players_stat = pd.DataFrame(columns = joined_columns)
    for team in teams:
        print('team '+ str(team))
        #time.sleep(0.5)
        team_roster = pd.read_pickle( root_path + 'data/rosters/'+season+'_roster_'+str(team)+'.pickle')
        #team_roster = CommonTeamRoster(team_id = team, season = season).get_data_frames()[0]
        players_season_stat = pd.DataFrame(columns = player_stat_columns)
        for player in team_roster['PLAYER_ID']:  
            print('     load for '+str(player))
            time.sleep(0.5)
            player_stat = PlayerCareerStats(player_id = player).get_data_frames()[0]
            player_stat = player_stat[player_stat['SEASON_ID']==season]
            #find the number of games played by each player to get mean stat:
            time.sleep(0.5)
            game_num = PlayerGameLog(player_id = player, season_all = season).get_data_frames()[0].shape[0]
            for f in to_mean_features:
                player_stat[f] = player_stat[f].map(lambda x: x / game_num)
            players_season_stat = players_season_stat.append(player_stat)

        players_season_stat = players_season_stat[players_season_stat['TEAM_ID'] == team]
        joined_players_stat = joined_players_stat.append(
            team_roster.merge(players_season_stat, how = 'inner', left_on='PLAYER_ID', right_on = 'PLAYER_ID', suffixes=('_l','_r'))
        )

    
    joined_players_stat.reset_index(drop=True ,inplace=True)
    dir_path = root_path + "/data/all_games_per_seasone/"
    serialize(joined_players_stat, dir_path=dir_path, filename='mean_players_stat_per' + season)
    
    return joined_players_stat


def window_team(team_id, features:list, data:pd.DataFrame, step = 3):
    '''
        Build samples consisting of mean stat of step num last games and vector of actual values for the 'stepth' game
    '''
    
    team_games = data[data['Team_ID'] == team_id]
    team_games = choose_features(features, team_games)

    mean_games = pd.DataFrame( [team_games[i:i + step].mean() for i in range(len(team_games) - step)] )
    step_games = team_games[step:] #actual stat of the 'stepth' game

    return [mean_games, step_games]

def get_map_window_games(team_id, data:pd.DataFrame, window = 3, f_to_mean = None):
    '''
        Maps current game stat with the mean of team's last games. The number of games is controlled by window param
        Returns dict { game_id: mean(last in-window games)}   
        Be careful: window size affects the games included      
    '''
    if f_to_mean is None:
        f_to_mean = features
    team_games = data[data['Team_ID'] == team_id]
    game_list = team_games['Game_ID'][window:]

    team_games = choose_features(f_to_mean, team_games)
    mean_games = [
        team_games[i:i + window].mean() for i in range(len(team_games) - window)
        ]

    dict_games = {
        game_id : games_mean for game_id, games_mean in zip(game_list, mean_games)
    }


    return dict_games
    

def serialize(data:pd.DataFrame, dir_path, filename):
    file_csv = dir_path + filename + ".csv"
    file_pickle = dir_path + filename + ".pickle"
    data.to_csv(file_csv, index=False)
    data.to_pickle(file_pickle)



def build_table():
    '''
        Downloads statistics of non-play-off games per specified seasons per all commands as dataframes.
        Per every game there is stat regarding both team. Dataframes are serialized and compiled to big one
        
    '''
    seasons = ['2008-09', '2009-10','2010-11','2011-12','2012-13','2013-14','2014-15','2015-16','2016-17', '2017-18', '2018-19' ]
    dir_path = "../data/all_games_per_seasone/"

    nba_teams = teams.get_teams()
    teams_fullnames_dict = {m['full_name']: m['id'] for m in nba_teams}

    train_data = pd.DataFrame()
    print("start")
    for season in seasons:
        season_data = pd.DataFrame()
        print(season)
        for name, team_id in teams_fullnames_dict.items():
            print(name)
            time.sleep(1)
            season_team_data = teamgamelog.TeamGameLog(team_id, season_all = season).get_data_frames()[0]
            season_data = season_data.append(season_team_data)

        season_data.sort_values(by = ['Game_ID'], inplace = True)
        season_data.reset_index(drop = True, inplace = True)        
        serialize(data = season_data, dir_path = dir_path, filename = season)
        train_data = train_data.append(season_data, )

    train_data.reset_index(drop = True, inplace = True)
    serialize(data = train_data, dir_path = dir_path, filename = "all_seasons")

    return train_data
    
def prepare_features(data:pd.DataFrame, drop_col = True, col_to_drop = None):
    '''
        Drops non-int fileds if drop_col = True
        Creates residual features: the difference between teams' stats per game
    '''
    if drop_col:
        
            col_to_drop = ['Team_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'W', 'L', 
            'W_PCT', 'MIN' ] if col_to_drop is None else col_to_drop
            data = data.drop(col_to_drop, axis = 1)

    data_difference = pd.DataFrame(columns=data.columns)

    for i in np.arange(0,len(data),2):
        left = data.iloc[i]
        right = data.iloc[i+1]
        data_difference = data_difference.append(left - right, ignore_index=True)

    return data_difference

def choose_features(features:list, data:pd.DataFrame):
    '''
        Chooses specified features from data
    '''
    all_features = data.columns
    features_to_drop = [f for f in all_features if f not in features]
    data = data.drop(features_to_drop, axis = 1)
    return data

def to_train(train_data:pd.DataFrame, ftrs = None):
    '''
        Prepares train samples from Data Frame
    '''
    if ftrs is None:
        ftrs = features
    train_data = prepare_features(train_data) 
    X_train = choose_features(ftrs, train_data).values.astype('float64')
    return X_train[:,:-1], X_train[:,-1]

def to_test(test_data:pd.DataFrame,  ftrs = None):
    if ftrs is None:
        ftrs = features
    test_data = prepare_features(test_data) 
    X_test = choose_features(ftrs, test_data).values.astype('float64')
    return X_test[:,:-1], X_test[:,-1]

def to_window_test(test_data:pd.DataFrame, window_games = None , ftrs = None, spec_game = None):
    '''
        Prepares test samples with window: 
            the difference between teams' are count based on mean of last several games. But PTS has actual value as it is it game
    '''
    if window_games is None:
        with open(root_path+"temp/window_games.pickle", 'rb') as f:
            window_games = pickle.load(f)
    
    if not spec_game is None:
        test_data = test_data[test_data['Game_ID'] == spec_game ]
    #test window - replace actual stat for game with mean stat
    test_data_window = pd.DataFrame(columns=test_data.columns)
    for game in test_data.iterrows():
        team_id = game[1]['Team_ID']
        game_id = game[1]['Game_ID']
        mean_game = window_games[team_id][game_id]
        mean_game['PTS'] = game[1]['PTS'] # mean prepared stats but actual result points
        test_data_window = test_data_window.append(mean_game, ignore_index=True)
 


    if ftrs is None:
        ftrs = features
    test_data_window = prepare_features(test_data_window, drop_col = False) 
    X_test_window = choose_features(ftrs, test_data_window).values.astype('float64')

    return X_test_window[:,:-1], X_test_window[:,-1]




def build_window_dict(data:pd.DataFrame,season='2018-19', save = False, type = None, f_to_mean = None, window = 5):
    '''
        Creates dict:
        { 
            team_id :{
                game_id : mean_window_game

            }           
        }

        where mean_window_game is a game, obtained with get_map_window_games function.
        Note, that stat has only chosen features
        
    '''
    window_games_dict = {}
    teams = set(data['Team_ID'])
    for team in teams:
        if type == "move_ave":
            window_games_dict[team] = get_dict_moving_ave_per_team(data, team, trg_label = 'PTS', f_to_mean = f_to_mean)
        else:
            window_games_dict[team] = get_map_window_games(team, data, window = window, f_to_mean = f_to_mean)

    if save:
        with open (root_path+"temp/window_games" + season +".pickle", 'wb+') as f:
            pickle.dump(window_games_dict, f)

    return window_games_dict 

def moving_ave_per_team(data:pd.DataFrame, team_id, trg_label = 'PTS', ftrs = None):
    '''
        Return moving avegare for the last game of specified team with actual game's PTS value
    '''

    team_data = data[data['Team_ID'] == team_id]
    if len(team_data) <  2:
        raise RuntimeError("Attempt to dip deep into the future")

    #if features is None:
    chosen_features = features if ftrs is None else ftrs
    team_data = choose_features(chosen_features, team_data)    
    mean_game = team_data[:-1].mean()
    act_game = team_data[-1:]
    mean_game[trg_label] = act_game[trg_label]
    return mean_game
    

def get_dict_moving_ave_per_team(data:pd.DataFrame, team_id, trg_label='PTS', f_to_mean = None):
    '''
        Create a dictionary, which map actual game with every game's stat is average of the last games except first. PTS remains actual
        Args:
            bound - start of games to get ave
    '''
    team_data = data[data['Team_ID'] == team_id]
    game_list = team_data['Game_ID'][1:]
    moving_ave = [ moving_ave_per_team(team_data[0:i], team_id, trg_label, ftrs = f_to_mean) for i in np.arange(2, len(team_data) + 1 ) ]

    dict_games = {
        game_id : games_mean for game_id, games_mean in zip(game_list, moving_ave)
    }
    return dict_games

def undercut_test(test_data:pd.DataFrame, window_games):
    '''
        Leave games presented in window_games dict
    '''
    undercut_data = pd.DataFrame(columns = test_data.columns)
    for game in test_data.iterrows():
       team_id = game[1]['Team_ID']
       game_id = game[1]['Game_ID']
       if game_id in window_games[team_id]:
           undercut_data = undercut_data.append(game[1], ignore_index = True)
    return undercut_data




def make_model_logistic_regr(x_train, y_train, x_test, y_test, x_test_w, y_test_w):
    y_train = np.sign(y_train)
    y_test = np.sign(y_test)
    y_test_w = np.sign(y_test_w)

    logis = linear_model.LogisticRegression(penalty='l2')
    logis.fit(x_train, y_train) 
    
    print('Train')
    print(logis.score(x_train,y_train))
    y_pred = logis.predict(x_train)
    work.check_model(y_pred,y_train)
    
    print('Test')
    print(logis.score(x_test, y_test))
    y_pred = logis.predict(x_test)
    work.check_model(y_pred,y_test)
    
    print('Window Test')
    print(logis.score(x_test_w, y_test_w))
    y_pred = logis.predict(x_test_w)
    work.check_model(y_pred,y_test_w)
    return logis

def estim_model_boosting(x_train, y_train, x_test, y_test, x_test_w, y_test_w, model):
    y_train = np.sign(y_train)
    y_test = np.sign(y_test)
    y_test_w = np.sign(y_test_w)
    print('Train')
    y_pred = model.predict(x_train)
    print(accuracy_score(y_train, y_pred))
    work.check_model(y_pred,y_train)
    print('Test')
    y_pred = model.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    work.check_model(y_pred,y_test)
    print('Window Test')
    y_pred = model.predict(x_test_w)
    print(accuracy_score(y_test_w, y_pred))
    work.check_model(y_pred,y_test_w)


def make_model_lasso_regr(x_train, y_train, x_test, y_test, x_test_w, y_test_w, alpha = 0.1):
    reg = linear_model.Lasso(alpha = alpha)
    reg.fit(x_train, y_train)
    print(reg.coef_)
    print(reg.intercept_)
    print('Train')
    print(reg.score(x_train,y_train))
    print(mean_absolute_error (y_train, reg.predict(x_train) ) )
    print('Test')
    print(reg.score(x_test, y_test))
    print(mean_absolute_error (y_test, reg.predict(x_test) ) )
    print('Window Test')
    print(reg.score(x_test_w, y_test_w))
    print(mean_absolute_error (y_test_w, reg.predict(x_test_w) ) )
    return reg

def make_model_lasso_boosting(x_train, y_train, x_test, y_test, x_test_w, y_test_w):
    y_train = np.sign(y_train)
    y_test = np.sign(y_test)
    y_test_w = np.sign(y_test_w)
    lgbm_pipeline = Pipeline([
        ('lgbm', lgb.LGBMClassifier(objective = 'binary', metric = 'binary_error'))
    ])

    k = 5
    skf = StratifiedKFold(n_splits=k, random_state=42)

    param_grid = {
        'lgbm__learning_rate':[0.1, 0.001, 0.01],
        'lgbm__n_estimators':[100, 200, 500],
        'lgbm__max_depth':[ 3,5,10,20],
    }

    grid = GridSearchCV(estimator=lgbm_pipeline,  param_grid=param_grid, cv=skf)
    grid_model = grid.fit(x_train, y_train)
    print(grid_model.best_params_)
    
    return grid


def main(dir_path = "data/all_games_per_seasone/"):
    dir_path = root_path + dir_path   
    season_data15 = pd.read_pickle(dir_path + '2015-16.pickle')  
    season_data16 = pd.read_pickle(dir_path + '2016-17.pickle')  
    season_data17 = pd.read_pickle(dir_path + '2017-18.pickle')  
    season_data18 = pd.read_pickle(dir_path + '2018-19.pickle')


    #work.build_window_dict(season_data18, season = '2018-19', save = True, f_to_mean=fts,type = 'move_ave')
    with open(root_path + "temp/window_games.pickle", 'rb') as f:
        window_games = pickle.load(f)


    train_data = season_data15
    train_data = train_data.append(season_data16)
    train_data = train_data.append(season_data17)

    test_data = undercut_test(season_data18, window_games)

    x_train, y_train = to_train(train_data)
    x_test, y_test = to_test(test_data)
    x_test_window, y_test_window = to_window_test(test_data, window_games=window_games)

    m = make_model_lasso_regr(x_train, y_train, x_test, y_test, x_test_w, y_test_w)




root_path = './'
features = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
       'PTS']
SEASON = '2018-19'

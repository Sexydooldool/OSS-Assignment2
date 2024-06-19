import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

def transform_and_cluster_to_csv(file_name, n_clusters=3):
    data = pd.read_csv(file_name, delimiter='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
    ratings_matrix = data.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
    ratings_array = ratings_matrix.to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(ratings_array)
    clusters_df = pd.DataFrame(clusters, index=ratings_matrix.index, columns=['Cluster'])
    output_file_name_matrix = file_name.replace('.dat', '_matrix.csv')
    output_file_name_clusters = file_name.replace('.dat', '_clusters.csv')
    ratings_matrix.to_csv(output_file_name_matrix)
    clusters_df.to_csv(output_file_name_clusters)
    return output_file_name_matrix, output_file_name_clusters

def separate_rating_matrix_by_cluster(matrix_file_name, cluster_file_name, n_clusters):
    ratings_matrix = pd.read_csv(matrix_file_name, index_col=0)
    clusters_df = pd.read_csv(cluster_file_name, index_col=0)
    cluster_files = []
    for cluster_label in range(n_clusters):
        cluster_users = clusters_df[clusters_df['Cluster'] == cluster_label].index
        cluster_ratings_matrix = ratings_matrix.loc[cluster_users]
        output_file_name = matrix_file_name.replace('.csv', f'_cluster_{cluster_label}.csv')
        cluster_ratings_matrix.to_csv(output_file_name)
        cluster_files.append(output_file_name)
    return cluster_files

def calculate_basic_statistics(cluster_ratings_matrix):
    statistics = defaultdict(list)
    for movie in cluster_ratings_matrix.columns:
        ratings = cluster_ratings_matrix[movie]
        non_zero_ratings = ratings[ratings > 0]
        average = non_zero_ratings.mean()
        additive_utilitarian = non_zero_ratings.sum()
        simple_count = non_zero_ratings.count()
        approval_voting = non_zero_ratings[non_zero_ratings >= 4].count()
        sorted_ratings = non_zero_ratings.sort_values(ascending=False)
        borda_count = sum((len(sorted_ratings) - rank) * rating for rank, rating in enumerate(sorted_ratings))
        statistics['MovieID'].append(movie)
        statistics['Average'].append(average)
        statistics['Additive Utilitarian'].append(additive_utilitarian)
        statistics['Simple Count'].append(simple_count)
        statistics['Approval Voting'].append(approval_voting)
        statistics['Borda Count'].append(borda_count)
    return pd.DataFrame(statistics)

def calculate_copeland_score(cluster_ratings_matrix):
    movies = cluster_ratings_matrix.columns
    num_movies = len(movies)
    approval_matrix = (cluster_ratings_matrix >= 4).astype(int)
    approval_sums = approval_matrix.sum(axis=0).values
    copeland_scores = np.zeros(num_movies)
    for i in range(num_movies):
        for j in range(num_movies):
            if i != j:
                if approval_sums[i] > approval_sums[j]:
                    copeland_scores[i] += 1
                elif approval_sums[i] < approval_sums[j]:
                    copeland_scores[i] -= 1
    return pd.DataFrame({'MovieID': movies, 'Copeland Rule': copeland_scores})

def calculate_statistics_for_clusters(cluster_files):
    recommendations = defaultdict(lambda: defaultdict(list))
    for cluster_file in cluster_files:
        cluster_ratings_matrix = pd.read_csv(cluster_file, index_col=0)
        basic_stats_df = calculate_basic_statistics(cluster_ratings_matrix)
        copeland_score_df = calculate_copeland_score(cluster_ratings_matrix)
        stats_df = pd.merge(basic_stats_df, copeland_score_df, on='MovieID')
        output_file_name = cluster_file.replace('.csv', '_stats.csv')
        stats_df.to_csv(output_file_name, index=False)
        criteria = ['Average', 'Additive Utilitarian', 'Simple Count', 'Approval Voting', 'Borda Count', 'Copeland Rule']
        top_10_movies = pd.DataFrame()
        for criterion in criteria:
            top_10 = stats_df.nlargest(10, criterion)
            top_10['Criterion'] = criterion
            top_10_movies = pd.concat([top_10_movies, top_10])
            cluster_num = int(cluster_file.split('_')[-1].split('.')[0])
            recommendations[cluster_num][criterion] = top_10['MovieID'].tolist()
        top_10_file_name = cluster_file.replace('.csv', '_top_60_movies.csv')
        top_10_movies.to_csv(top_10_file_name, index=False)
        print(f"Top 60 movies for {cluster_file} saved to {top_10_file_name}")
    return recommendations

file_name = 'ratings.dat'
n_clusters = 3
matrix_file, cluster_file = transform_and_cluster_to_csv(file_name, n_clusters)
cluster_files = separate_rating_matrix_by_cluster(matrix_file, cluster_file, n_clusters)
recommendations = calculate_statistics_for_clusters(cluster_files)

for cluster, recs in recommendations.items():
    for criterion, movies in recs.items():
        print(f"Cluster {cluster}, Criterion {criterion}: {movies}")

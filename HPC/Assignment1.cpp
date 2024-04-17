#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Define the graph structure (undirected graph)
class Graph
{
    int V;                   // Number of vertices
    vector<vector<int>> adj; // Adjacency list

public:
    Graph(int V)
    {
        this->V = V;
        adj.resize(V);
    }

    // Function to add an edge to the graph
    void addEdge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Parallel Breadth First Search
    // Parallel Breadth First Search
    void parallelBFS(int source)
    {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[source] = true;
        q.push(source);

        while (!q.empty())
        {
#pragma omp parallel shared(q, visited)
            {
#pragma omp for
                for (int i = 0; i < q.size(); ++i)
                {
                    int u;
#pragma omp critical
                    {
                        u = q.front();
                        q.pop();
                    }
                    cout << u << " ";

                    // Enqueue adjacent vertices of the dequeued vertex
                    for (int j = 0; j < adj[u].size(); ++j)
                    {
                        int v = adj[u][j];
                        if (!visited[v])
                        {
#pragma omp critical
                            {
                                visited[v] = true;
                                q.push(v);
                            }
                        }
                    }
                }
            }
        }
    }

    // Parallel Depth First Search
    // Parallel Depth First Search
    void parallelDFSUtil(int v, vector<bool> &visited)
    {
        visited[v] = true;
        cout << v << " ";

// Traverse all adjacent vertices
#pragma omp parallel for
        for (int i = 0; i < adj[v].size(); ++i)
        {
            int u = adj[v][i];
            if (!visited[u])
            {
                parallelDFSUtil(u, visited);
            }
        }
    }

    void parallelDFS(int source)
    {
        vector<bool> visited(V, false);
        parallelDFSUtil(source, visited);
    }
};

int main()
{
    // Create a graph
    Graph g(6);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 4);
    g.addEdge(3, 5);
    g.addEdge(4, 5);

    cout << "Parallel Breadth First Search (BFS) starting from vertex 0: ";
    g.parallelBFS(0);
    cout << endl;

    cout << "Parallel Depth First Search (DFS) starting from vertex 0: ";
    g.parallelDFS(0);
    cout << endl;

    return 0;
}

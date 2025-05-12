#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

const int STATES = 12;
double gamma = 0.75;
double alpha = 0.9;

map<string, int> location_to_state = {
	{"A", 0}, {"B", 1}, {"C", 2}, {"D", 3}, {"E", 4}, {"F", 5}, {"G", 6}, {"H", 7}, {"I", 8}, {"J", 9}, {"K", 10}, {"L", 11}};
vector<string> state_to_location = {
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"};

// Reward matrix
typedef vector<vector<int>> MatrixInt;
MatrixInt R = {
	{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},
	{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
	{0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
	{0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0},
	{0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1},
	{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0},
	{0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0},
	{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
	{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0}};

vector<string> route(const string &start, const string &end)
{
	// Copy reward matrix and set goal
	MatrixInt R_new = R;
	int end_state = location_to_state[end];
	R_new[end_state][end_state] = 1000;

	// Initialize Q matrix
	vector<vector<double>> Q(STATES, vector<double>(STATES, 0.0));

	// Q-learning
	for (int i = 0; i < 1000; ++i)
	{
		int curr = rand() % STATES;
		vector<int> actions;
		for (int j = 0; j < STATES; ++j)
		{
			if (R_new[curr][j] > 0)
				actions.push_back(j);
		}
		if (actions.empty())
			continue;
		int next = actions[rand() % actions.size()];
		double td = R_new[curr][next] +
					gamma * Q[next][max_element(Q[next].begin(), Q[next].end()) - Q[next].begin()] - Q[curr][next];
		Q[curr][next] += alpha * td;
	}

	// Build route
	vector<string> path;
	path.push_back(start);
	string next_loc = start;
	while (next_loc != end)
	{
		int state = location_to_state[next_loc];
		int best_next = max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
		next_loc = state_to_location[best_next];
		path.push_back(next_loc);
	}
	return path;
}

vector<string> best_route(const string &start, const string &mid, const string &end)
{
	vector<string> first = route(start, mid);
	vector<string> second = route(mid, end);
	// Combine, avoid duplicate mid
	first.insert(first.end(), second.begin() + 1, second.end());
	return first;
}

int main()
{
	srand(static_cast<unsigned>(time(0)));
	vector<string> result = best_route("E", "K", "G");
	cout << "Route: ";
	for (const auto &loc : result)
	{
		cout << loc;
		if (&loc != &result.back())
			cout << " -> ";
	}
	cout << endl;
	return 0;
}

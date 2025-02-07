#include<iostream>
#include<string>
using namespace std;
int dp[101][1000001];
bool checkSubString(string s, string t){
    int n = s.length(), m = t.length();
    for(int i = 0; i <= max(n, m); i ++){
        dp[i][0] = 0;
        dp[0][i] = 0;
    }
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= m; j ++){
            dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            if(s[i-1] == t[j-1])dp[i][j] = max(dp[i][j], dp[i-1][j-1]+1);
        }
    }
    for(int j = 1; j <= m; j ++){
        if(dp[n][j] == n)return true;
    }
    return false;
}
int main(){
    string s, t;
    cin >> s >> t;
    if(checkSubString(s, t))cout << "true";
    else cout << "false";
    return 0;
}
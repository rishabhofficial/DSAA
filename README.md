# DSA Interview Revision Notes (C++11)

## Table of Contents
1. [Array Techniques](#array-techniques)
2. [String Techniques](#string-techniques)
3. [Hash Table](#hash-table)
4. [Linked List](#linked-list)
5. [Stack & Queue](#stack--queue)
6. [Recursion & Backtracking](#recursion--backtracking)
7. [Sorting Algorithms](#sorting-algorithms)
8. [Binary Search](#binary-search)
9. [Two Pointers](#two-pointers)
10. [Sliding Window](#sliding-window)
11. [Matrix](#matrix)
12. [Tree](#tree)
13. [Graph](#graph)
14. [Heap / Priority Queue](#heap--priority-queue)
15. [Trie](#trie)
16. [Dynamic Programming](#dynamic-programming)
17. [Intervals](#intervals)
18. [Geometry](#geometry)
19. [Bit Manipulation](#bit-manipulation)
20. [Disjoint Set Union (DSU)](#disjoint-set-union-dsu)
21. [Monotonic Stack](#monotonic-stack)
22. [Notable Algorithms](#notable-algorithms)

---

## Array Techniques

### Key Points
- Traversing array twice/thrice is still O(n)
- Index as hash key for O(1) space when values are 1 to N
- Precomputation with prefix/suffix sums
- Consider sorting if order doesn't matter

### Prefix Sum
```cpp
#include <vector>
using namespace std;

// Build prefix sum array
vector<int> buildPrefixSum(const vector<int>& arr) {
    int n = arr.size();
    vector<int> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) {
        prefix[i + 1] = prefix[i] + arr[i];
    }
    return prefix;
}

// Get sum of range [l, r] (inclusive)
int rangeSum(const vector<int>& prefix, int l, int r) {
    return prefix[r + 1] - prefix[l];
}
```

### Kadane's Algorithm (Maximum Subarray Sum)
```cpp
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;

int maxSubArraySum(const vector<int>& arr) {
    if (arr.empty()) return 0;

    int maxCurrent = arr[0];
    int maxGlobal = arr[0];

    for (int i = 1; i < arr.size(); i++) {
        maxCurrent = max(arr[i], maxCurrent + arr[i]);
        maxGlobal = max(maxGlobal, maxCurrent);
    }

    return maxGlobal;
}

// With indices tracking
struct Result {
    int sum;
    int start;
    int end;
};

Result maxSubArrayWithIndices(const vector<int>& arr) {
    if (arr.empty()) return {0, -1, -1};

    int maxCurrent = arr[0];
    int maxGlobal = arr[0];
    int start = 0, end = 0, tempStart = 0;

    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] > maxCurrent + arr[i]) {
            maxCurrent = arr[i];
            tempStart = i;
        } else {
            maxCurrent += arr[i];
        }

        if (maxCurrent > maxGlobal) {
            maxGlobal = maxCurrent;
            start = tempStart;
            end = i;
        }
    }

    return {maxGlobal, start, end};
}
```

### Dutch National Flag (Sort Colors)
```cpp
#include <vector>
using namespace std;

void sortColors(vector<int>& nums) {
    int low = 0, mid = 0, high = nums.size() - 1;

    while (mid <= high) {
        if (nums[mid] == 0) {
            swap(nums[low++], nums[mid++]);
        } else if (nums[mid] == 1) {
            mid++;
        } else {
            swap(nums[mid], nums[high--]);
        }
    }
}
```

### Next Permutation
```cpp
#include <vector>
#include <algorithm>
using namespace std;

void nextPermutation(vector<int>& nums) {
    int n = nums.size();
    int i = n - 2;

    // Find first decreasing element from right
    while (i >= 0 && nums[i] >= nums[i + 1]) {
        i--;
    }

    if (i >= 0) {
        // Find element just larger than nums[i]
        int j = n - 1;
        while (nums[j] <= nums[i]) {
            j--;
        }
        swap(nums[i], nums[j]);
    }

    // Reverse the suffix
    reverse(nums.begin() + i + 1, nums.end());
}
```

---

## String Techniques

### Key Points
- Character counting uses O(1) space (26 letters max)
- Anagram detection: sort or frequency count
- Palindrome: two pointers from ends or expand from center

### Character Frequency Count
```cpp
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

// Using array (for lowercase letters only)
vector<int> countChars(const string& s) {
    vector<int> freq(26, 0);
    for (char c : s) {
        freq[c - 'a']++;
    }
    return freq;
}

// Using hash map (for any characters)
unordered_map<char, int> countCharsMap(const string& s) {
    unordered_map<char, int> freq;
    for (char c : s) {
        freq[c]++;
    }
    return freq;
}
```

### Anagram Check
```cpp
#include <string>
#include <algorithm>
using namespace std;

bool isAnagram(string s1, string s2) {
    if (s1.length() != s2.length()) return false;

    // Method 1: Sort
    sort(s1.begin(), s1.end());
    sort(s2.begin(), s2.end());
    return s1 == s2;
}

bool isAnagramFreq(const string& s1, const string& s2) {
    if (s1.length() != s2.length()) return false;

    // Method 2: Frequency count
    vector<int> freq(26, 0);
    for (int i = 0; i < s1.length(); i++) {
        freq[s1[i] - 'a']++;
        freq[s2[i] - 'a']--;
    }

    for (int f : freq) {
        if (f != 0) return false;
    }
    return true;
}
```

### Palindrome Check
```cpp
#include <string>
using namespace std;

bool isPalindrome(const string& s) {
    int left = 0, right = s.length() - 1;
    while (left < right) {
        if (s[left] != s[right]) return false;
        left++;
        right--;
    }
    return true;
}

// Expand around center (for finding palindromic substrings)
int expandAroundCenter(const string& s, int left, int right) {
    while (left >= 0 && right < s.length() && s[left] == s[right]) {
        left--;
        right++;
    }
    return right - left - 1;
}

string longestPalindrome(const string& s) {
    if (s.empty()) return "";

    int start = 0, maxLen = 1;

    for (int i = 0; i < s.length(); i++) {
        int len1 = expandAroundCenter(s, i, i);     // Odd length
        int len2 = expandAroundCenter(s, i, i + 1); // Even length
        int len = max(len1, len2);

        if (len > maxLen) {
            maxLen = len;
            start = i - (len - 1) / 2;
        }
    }

    return s.substr(start, maxLen);
}
```

### KMP Pattern Matching
```cpp
#include <string>
#include <vector>
using namespace std;

vector<int> computeLPS(const string& pattern) {
    int m = pattern.length();
    vector<int> lps(m, 0);
    int len = 0, i = 1;

    while (i < m) {
        if (pattern[i] == pattern[len]) {
            lps[i++] = ++len;
        } else if (len != 0) {
            len = lps[len - 1];
        } else {
            lps[i++] = 0;
        }
    }
    return lps;
}

vector<int> KMPSearch(const string& text, const string& pattern) {
    vector<int> result;
    int n = text.length(), m = pattern.length();
    vector<int> lps = computeLPS(pattern);

    int i = 0, j = 0;
    while (i < n) {
        if (text[i] == pattern[j]) {
            i++;
            j++;
        }

        if (j == m) {
            result.push_back(i - j);
            j = lps[j - 1];
        } else if (i < n && text[i] != pattern[j]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    return result;
}
```

---

## Hash Table

### LRU Cache
```cpp
#include <unordered_map>
#include <list>
using namespace std;

class LRUCache {
private:
    int capacity;
    list<pair<int, int>> cache;  // {key, value}
    unordered_map<int, list<pair<int, int>>::iterator> map;

    void moveToFront(int key, int value) {
        cache.erase(map[key]);
        cache.push_front({key, value});
        map[key] = cache.begin();
    }

public:
    LRUCache(int capacity) : capacity(capacity) {}

    int get(int key) {
        if (map.find(key) == map.end()) return -1;

        int value = map[key]->second;
        moveToFront(key, value);
        return value;
    }

    void put(int key, int value) {
        if (map.find(key) != map.end()) {
            moveToFront(key, value);
            return;
        }

        if (cache.size() >= capacity) {
            int lruKey = cache.back().first;
            cache.pop_back();
            map.erase(lruKey);
        }

        cache.push_front({key, value});
        map[key] = cache.begin();
    }
};
```

### LFU Cache
```cpp
#include <unordered_map>
#include <list>
#include <climits>
using namespace std;

class LFUCache {
private:
    int capacity, minFreq;
    unordered_map<int, pair<int, int>> keyToValFreq;  // key -> {value, freq}
    unordered_map<int, list<int>> freqToKeys;         // freq -> list of keys
    unordered_map<int, list<int>::iterator> keyToIter; // key -> iterator in list

public:
    LFUCache(int capacity) : capacity(capacity), minFreq(0) {}

    int get(int key) {
        if (keyToValFreq.find(key) == keyToValFreq.end()) return -1;

        int value = keyToValFreq[key].first;
        int freq = keyToValFreq[key].second;

        // Remove from current frequency list
        freqToKeys[freq].erase(keyToIter[key]);
        if (freqToKeys[freq].empty()) {
            freqToKeys.erase(freq);
            if (minFreq == freq) minFreq++;
        }

        // Add to next frequency list
        freq++;
        keyToValFreq[key].second = freq;
        freqToKeys[freq].push_front(key);
        keyToIter[key] = freqToKeys[freq].begin();

        return value;
    }

    void put(int key, int value) {
        if (capacity <= 0) return;

        if (keyToValFreq.find(key) != keyToValFreq.end()) {
            keyToValFreq[key].first = value;
            get(key);  // Update frequency
            return;
        }

        if (keyToValFreq.size() >= capacity) {
            int evictKey = freqToKeys[minFreq].back();
            freqToKeys[minFreq].pop_back();
            if (freqToKeys[minFreq].empty()) {
                freqToKeys.erase(minFreq);
            }
            keyToValFreq.erase(evictKey);
            keyToIter.erase(evictKey);
        }

        minFreq = 1;
        keyToValFreq[key] = {value, 1};
        freqToKeys[1].push_front(key);
        keyToIter[key] = freqToKeys[1].begin();
    }
};
```

---

## Linked List

### Key Points
- Use dummy/sentinel nodes to handle edge cases
- Two pointers: fast/slow for cycle detection, k-distance for kth from end
- Reverse in-place to save space

### Basic Node Structure
```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* n) : val(x), next(n) {}
};
```

### Reverse Linked List
```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;

    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }

    return prev;
}

// Recursive
ListNode* reverseListRecursive(ListNode* head) {
    if (!head || !head->next) return head;

    ListNode* newHead = reverseListRecursive(head->next);
    head->next->next = head;
    head->next = nullptr;

    return newHead;
}
```

### Detect Cycle (Floyd's Algorithm)
```cpp
bool hasCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;

    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }

    return false;
}

// Find cycle start
ListNode* detectCycleStart(ListNode* head) {
    ListNode *slow = head, *fast = head;

    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;

        if (slow == fast) {
            slow = head;
            while (slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }
    }

    return nullptr;
}
```

### Find Middle
```cpp
ListNode* findMiddle(ListNode* head) {
    ListNode *slow = head, *fast = head;

    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }

    return slow;
}
```

### Merge Two Sorted Lists
```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode dummy(0);
    ListNode* curr = &dummy;

    while (l1 && l2) {
        if (l1->val < l2->val) {
            curr->next = l1;
            l1 = l1->next;
        } else {
            curr->next = l2;
            l2 = l2->next;
        }
        curr = curr->next;
    }

    curr->next = l1 ? l1 : l2;
    return dummy.next;
}
```

### Kth Node from End
```cpp
ListNode* findKthFromEnd(ListNode* head, int k) {
    ListNode *fast = head, *slow = head;

    // Move fast k nodes ahead
    for (int i = 0; i < k; i++) {
        if (!fast) return nullptr;
        fast = fast->next;
    }

    while (fast) {
        slow = slow->next;
        fast = fast->next;
    }

    return slow;
}
```

---

## Stack & Queue

### Valid Parentheses
```cpp
#include <stack>
#include <string>
#include <unordered_map>
using namespace std;

bool isValid(const string& s) {
    stack<char> st;
    unordered_map<char, char> pairs = {{')', '('}, {']', '['}, {'}', '{'}};

    for (char c : s) {
        if (pairs.find(c) != pairs.end()) {
            if (st.empty() || st.top() != pairs[c]) return false;
            st.pop();
        } else {
            st.push(c);
        }
    }

    return st.empty();
}
```

### Min Stack
```cpp
#include <stack>
using namespace std;

class MinStack {
private:
    stack<int> st;
    stack<int> minSt;

public:
    void push(int val) {
        st.push(val);
        if (minSt.empty() || val <= minSt.top()) {
            minSt.push(val);
        }
    }

    void pop() {
        if (st.top() == minSt.top()) {
            minSt.pop();
        }
        st.pop();
    }

    int top() {
        return st.top();
    }

    int getMin() {
        return minSt.top();
    }
};
```

### Implement Queue using Stacks
```cpp
#include <stack>
using namespace std;

class MyQueue {
private:
    stack<int> input, output;

    void transfer() {
        if (output.empty()) {
            while (!input.empty()) {
                output.push(input.top());
                input.pop();
            }
        }
    }

public:
    void push(int x) {
        input.push(x);
    }

    int pop() {
        transfer();
        int val = output.top();
        output.pop();
        return val;
    }

    int peek() {
        transfer();
        return output.top();
    }

    bool empty() {
        return input.empty() && output.empty();
    }
};
```

---

## Recursion & Backtracking

### Key Points
- Always define base case
- Recursion uses O(n) stack space unless tail-call optimized
- For permutations/combinations: use backtracking

### Permutations
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> result;

    void backtrack(vector<int>& nums, int start) {
        if (start == nums.size()) {
            result.push_back(nums);
            return;
        }

        for (int i = start; i < nums.size(); i++) {
            swap(nums[start], nums[i]);
            backtrack(nums, start + 1);
            swap(nums[start], nums[i]);  // Backtrack
        }
    }

    vector<vector<int>> permute(vector<int>& nums) {
        backtrack(nums, 0);
        return result;
    }
};
```

### Combinations
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> result;

    void backtrack(int n, int k, int start, vector<int>& current) {
        if (current.size() == k) {
            result.push_back(current);
            return;
        }

        for (int i = start; i <= n; i++) {
            current.push_back(i);
            backtrack(n, k, i + 1, current);
            current.pop_back();  // Backtrack
        }
    }

    vector<vector<int>> combine(int n, int k) {
        vector<int> current;
        backtrack(n, k, 1, current);
        return result;
    }
};
```

### Subsets
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> result;

    void backtrack(vector<int>& nums, int start, vector<int>& current) {
        result.push_back(current);

        for (int i = start; i < nums.size(); i++) {
            current.push_back(nums[i]);
            backtrack(nums, i + 1, current);
            current.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        vector<int> current;
        backtrack(nums, 0, current);
        return result;
    }
};
```

### N-Queens
```cpp
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    vector<vector<string>> result;

    bool isValid(vector<string>& board, int row, int col, int n) {
        // Check column
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') return false;
        }

        // Check upper-left diagonal
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') return false;
        }

        // Check upper-right diagonal
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') return false;
        }

        return true;
    }

    void backtrack(vector<string>& board, int row, int n) {
        if (row == n) {
            result.push_back(board);
            return;
        }

        for (int col = 0; col < n; col++) {
            if (isValid(board, row, col, n)) {
                board[row][col] = 'Q';
                backtrack(board, row + 1, n);
                board[row][col] = '.';
            }
        }
    }

    vector<vector<string>> solveNQueens(int n) {
        vector<string> board(n, string(n, '.'));
        backtrack(board, 0, n);
        return result;
    }
};
```

---

## Sorting Algorithms

### Quick Sort
```cpp
#include <vector>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }

    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

### Merge Sort
```cpp
#include <vector>
using namespace std;

void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int i = 0; i < temp.size(); i++) {
        arr[left + i] = temp[i];
    }
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
```

### Heap Sort
```cpp
#include <vector>
using namespace std;

void heapify(vector<int>& arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest]) largest = left;
    if (right < n && arr[right] > arr[largest]) largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(vector<int>& arr) {
    int n = arr.size();

    // Build max heap
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // Extract elements from heap
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}
```

### Counting Sort
```cpp
#include <vector>
using namespace std;

void countingSort(vector<int>& arr, int maxVal) {
    vector<int> count(maxVal + 1, 0);

    for (int num : arr) {
        count[num]++;
    }

    int idx = 0;
    for (int i = 0; i <= maxVal; i++) {
        while (count[i] > 0) {
            arr[idx++] = i;
            count[i]--;
        }
    }
}
```

---

## Binary Search

### Key Points
- Works on sorted arrays
- Time: O(log n)
- Watch for overflow: use `mid = left + (right - left) / 2`

### Standard Binary Search
```cpp
#include <vector>
using namespace std;

int binarySearch(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;  // Not found
}
```

### Lower Bound (First >= target)
```cpp
int lowerBound(const vector<int>& arr, int target) {
    int left = 0, right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}
```

### Upper Bound (First > target)
```cpp
int upperBound(const vector<int>& arr, int target) {
    int left = 0, right = arr.size();

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}
```

### Search in Rotated Sorted Array
```cpp
int searchRotated(const vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (nums[mid] == target) return mid;

        // Left half is sorted
        if (nums[left] <= nums[mid]) {
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        // Right half is sorted
        else {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    return -1;
}
```

### Binary Search on Answer
```cpp
// Example: Find minimum capacity to ship packages in D days
int shipWithinDays(vector<int>& weights, int days) {
    int left = *max_element(weights.begin(), weights.end());
    int right = accumulate(weights.begin(), weights.end(), 0);

    auto canShip = [&](int capacity) {
        int daysNeeded = 1, currentLoad = 0;
        for (int w : weights) {
            if (currentLoad + w > capacity) {
                daysNeeded++;
                currentLoad = 0;
            }
            currentLoad += w;
        }
        return daysNeeded <= days;
    };

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canShip(mid)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return left;
}
```

---

## Two Pointers

### Template
```cpp
// Opposite direction pointers (sorted array)
int left = 0, right = arr.size() - 1;
while (left < right) {
    // Process arr[left] and arr[right]
    // Adjust left++ or right-- based on condition
}

// Same direction pointers
int slow = 0, fast = 0;
while (fast < arr.size()) {
    // Process based on condition
    // Adjust slow and fast
}
```

### Two Sum (Sorted Array)
```cpp
#include <vector>
using namespace std;

vector<int> twoSumSorted(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left < right) {
        int sum = arr[left] + arr[right];

        if (sum == target) {
            return {left, right};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }

    return {};  // Not found
}
```

### Container With Most Water
```cpp
int maxArea(vector<int>& height) {
    int left = 0, right = height.size() - 1;
    int maxWater = 0;

    while (left < right) {
        int h = min(height[left], height[right]);
        int w = right - left;
        maxWater = max(maxWater, h * w);

        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }

    return maxWater;
}
```

### 3Sum
```cpp
#include <vector>
#include <algorithm>
using namespace std;

vector<vector<int>> threeSum(vector<int>& nums) {
    vector<vector<int>> result;
    sort(nums.begin(), nums.end());
    int n = nums.size();

    for (int i = 0; i < n - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;  // Skip duplicates

        int left = i + 1, right = n - 1;

        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];

            if (sum == 0) {
                result.push_back({nums[i], nums[left], nums[right]});

                while (left < right && nums[left] == nums[left + 1]) left++;
                while (left < right && nums[right] == nums[right - 1]) right--;

                left++;
                right--;
            } else if (sum < 0) {
                left++;
            } else {
                right--;
            }
        }
    }

    return result;
}
```

### Remove Duplicates from Sorted Array
```cpp
int removeDuplicates(vector<int>& nums) {
    if (nums.empty()) return 0;

    int slow = 0;
    for (int fast = 1; fast < nums.size(); fast++) {
        if (nums[fast] != nums[slow]) {
            slow++;
            nums[slow] = nums[fast];
        }
    }

    return slow + 1;
}
```

---

## Sliding Window

### Template
```cpp
// Fixed size window
int start = 0;
for (int end = 0; end < arr.size(); end++) {
    // Add arr[end] to window

    if (end - start + 1 == k) {  // Window size reached
        // Process window
        // Remove arr[start] from window
        start++;
    }
}

// Variable size window
int start = 0;
for (int end = 0; end < arr.size(); end++) {
    // Add arr[end] to window

    while (/* window invalid condition */) {
        // Remove arr[start] from window
        start++;
    }

    // Process valid window
}
```

### Maximum Sum Subarray of Size K
```cpp
int maxSumSubarray(const vector<int>& arr, int k) {
    int maxSum = INT_MIN;
    int currentSum = 0;
    int start = 0;

    for (int end = 0; end < arr.size(); end++) {
        currentSum += arr[end];

        if (end - start + 1 == k) {
            maxSum = max(maxSum, currentSum);
            currentSum -= arr[start];
            start++;
        }
    }

    return maxSum;
}
```

### Longest Substring Without Repeating Characters
```cpp
#include <string>
#include <unordered_set>
using namespace std;

int lengthOfLongestSubstring(const string& s) {
    unordered_set<char> chars;
    int maxLen = 0;
    int start = 0;

    for (int end = 0; end < s.length(); end++) {
        while (chars.count(s[end])) {
            chars.erase(s[start]);
            start++;
        }
        chars.insert(s[end]);
        maxLen = max(maxLen, end - start + 1);
    }

    return maxLen;
}
```

### Minimum Window Substring
```cpp
#include <string>
#include <unordered_map>
#include <climits>
using namespace std;

string minWindow(const string& s, const string& t) {
    unordered_map<char, int> need, have;

    for (char c : t) need[c]++;

    int required = need.size();
    int formed = 0;
    int minLen = INT_MAX;
    int minStart = 0;
    int start = 0;

    for (int end = 0; end < s.length(); end++) {
        char c = s[end];
        have[c]++;

        if (need.count(c) && have[c] == need[c]) {
            formed++;
        }

        while (formed == required) {
            if (end - start + 1 < minLen) {
                minLen = end - start + 1;
                minStart = start;
            }

            char leftChar = s[start];
            have[leftChar]--;

            if (need.count(leftChar) && have[leftChar] < need[leftChar]) {
                formed--;
            }

            start++;
        }
    }

    return minLen == INT_MAX ? "" : s.substr(minStart, minLen);
}
```

---

## Matrix

### Create Matrix
```cpp
#include <vector>
using namespace std;

// Create m x n matrix initialized with value
vector<vector<int>> matrix(m, vector<int>(n, 0));
```

### Transpose Matrix
```cpp
vector<vector<int>> transpose(const vector<vector<int>>& matrix) {
    int m = matrix.size(), n = matrix[0].size();
    vector<vector<int>> result(n, vector<int>(m));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}
```

### Rotate Matrix 90 Degrees Clockwise
```cpp
void rotate(vector<vector<int>>& matrix) {
    int n = matrix.size();

    // Transpose
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            swap(matrix[i][j], matrix[j][i]);
        }
    }

    // Reverse each row
    for (int i = 0; i < n; i++) {
        reverse(matrix[i].begin(), matrix[i].end());
    }
}
```

### Spiral Matrix Traversal
```cpp
vector<int> spiralOrder(const vector<vector<int>>& matrix) {
    vector<int> result;
    if (matrix.empty()) return result;

    int top = 0, bottom = matrix.size() - 1;
    int left = 0, right = matrix[0].size() - 1;

    while (top <= bottom && left <= right) {
        // Right
        for (int i = left; i <= right; i++) {
            result.push_back(matrix[top][i]);
        }
        top++;

        // Down
        for (int i = top; i <= bottom; i++) {
            result.push_back(matrix[i][right]);
        }
        right--;

        // Left
        if (top <= bottom) {
            for (int i = right; i >= left; i--) {
                result.push_back(matrix[bottom][i]);
            }
            bottom--;
        }

        // Up
        if (left <= right) {
            for (int i = bottom; i >= top; i--) {
                result.push_back(matrix[i][left]);
            }
            left++;
        }
    }

    return result;
}
```

### Set Matrix Zeroes (O(1) space)
```cpp
void setZeroes(vector<vector<int>>& matrix) {
    int m = matrix.size(), n = matrix[0].size();
    bool firstRowZero = false, firstColZero = false;

    // Check first row and column
    for (int j = 0; j < n; j++) {
        if (matrix[0][j] == 0) firstRowZero = true;
    }
    for (int i = 0; i < m; i++) {
        if (matrix[i][0] == 0) firstColZero = true;
    }

    // Mark zeros in first row/column
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (matrix[i][j] == 0) {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }

    // Set zeros based on marks
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                matrix[i][j] = 0;
            }
        }
    }

    // Handle first row and column
    if (firstRowZero) {
        for (int j = 0; j < n; j++) matrix[0][j] = 0;
    }
    if (firstColZero) {
        for (int i = 0; i < m; i++) matrix[i][0] = 0;
    }
}
```

---

## Tree

### Node Structure
```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
```

### Traversals

#### Recursive
```cpp
#include <vector>
using namespace std;

// Inorder: Left -> Root -> Right
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);
    inorder(root->right, result);
}

// Preorder: Root -> Left -> Right
void preorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    result.push_back(root->val);
    preorder(root->left, result);
    preorder(root->right, result);
}

// Postorder: Left -> Right -> Root
void postorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    postorder(root->left, result);
    postorder(root->right, result);
    result.push_back(root->val);
}
```

#### Iterative
```cpp
#include <stack>
#include <vector>
using namespace std;

vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;

    while (curr || !st.empty()) {
        while (curr) {
            st.push(curr);
            curr = curr->left;
        }
        curr = st.top();
        st.pop();
        result.push_back(curr->val);
        curr = curr->right;
    }

    return result;
}

vector<int> preorderIterative(TreeNode* root) {
    vector<int> result;
    if (!root) return result;

    stack<TreeNode*> st;
    st.push(root);

    while (!st.empty()) {
        TreeNode* node = st.top();
        st.pop();
        result.push_back(node->val);

        if (node->right) st.push(node->right);
        if (node->left) st.push(node->left);
    }

    return result;
}

vector<int> postorderIterative(TreeNode* root) {
    vector<int> result;
    if (!root) return result;

    stack<TreeNode*> st;
    TreeNode* lastVisited = nullptr;

    while (root || !st.empty()) {
        while (root) {
            st.push(root);
            root = root->left;
        }

        TreeNode* peekNode = st.top();

        if (!peekNode->right || peekNode->right == lastVisited) {
            result.push_back(peekNode->val);
            lastVisited = st.top();
            st.pop();
        } else {
            root = peekNode->right;
        }
    }

    return result;
}
```

### Level Order Traversal (BFS)
```cpp
#include <queue>
#include <vector>
using namespace std;

vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> level;

        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);

            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }

        result.push_back(level);
    }

    return result;
}
```

### Maximum Depth
```cpp
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
```

### Validate BST
```cpp
#include <climits>
using namespace std;

bool isValidBST(TreeNode* root, long minVal = LONG_MIN, long maxVal = LONG_MAX) {
    if (!root) return true;

    if (root->val <= minVal || root->val >= maxVal) return false;

    return isValidBST(root->left, minVal, root->val) &&
           isValidBST(root->right, root->val, maxVal);
}
```

### Lowest Common Ancestor (LCA)
```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;

    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);

    if (left && right) return root;
    return left ? left : right;
}

// For BST
TreeNode* lowestCommonAncestorBST(TreeNode* root, TreeNode* p, TreeNode* q) {
    while (root) {
        if (p->val < root->val && q->val < root->val) {
            root = root->left;
        } else if (p->val > root->val && q->val > root->val) {
            root = root->right;
        } else {
            return root;
        }
    }
    return nullptr;
}
```

### Serialize and Deserialize
```cpp
#include <string>
#include <sstream>
#include <queue>
using namespace std;

class Codec {
public:
    string serialize(TreeNode* root) {
        if (!root) return "null";

        string result;
        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();

            if (node) {
                result += to_string(node->val) + ",";
                q.push(node->left);
                q.push(node->right);
            } else {
                result += "null,";
            }
        }

        return result;
    }

    TreeNode* deserialize(string data) {
        if (data == "null") return nullptr;

        stringstream ss(data);
        string item;
        getline(ss, item, ',');

        TreeNode* root = new TreeNode(stoi(item));
        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();

            if (getline(ss, item, ',') && item != "null") {
                node->left = new TreeNode(stoi(item));
                q.push(node->left);
            }

            if (getline(ss, item, ',') && item != "null") {
                node->right = new TreeNode(stoi(item));
                q.push(node->right);
            }
        }

        return root;
    }
};
```

---

## Graph

### Graph Representations
```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>
using namespace std;

// Adjacency List
vector<vector<int>> adjList(n);  // n nodes
adjList[u].push_back(v);  // Add edge u -> v

// Adjacency List with weights
vector<vector<pair<int, int>>> adjListWeighted(n);
adjListWeighted[u].push_back({v, weight});

// Hash Map of Hash Maps
unordered_map<int, unordered_map<int, int>> graph;
graph[u][v] = weight;  // Add edge u -> v with weight

// Build from edge list
void buildGraph(vector<vector<int>>& edges, bool directed = false) {
    unordered_map<int, vector<int>> graph;
    for (auto& edge : edges) {
        int u = edge[0], v = edge[1];
        graph[u].push_back(v);
        if (!directed) graph[v].push_back(u);
    }
}
```

### DFS
```cpp
#include <vector>
#include <unordered_set>
using namespace std;

// Recursive DFS
void dfs(int node, vector<vector<int>>& graph, unordered_set<int>& visited) {
    if (visited.count(node)) return;
    visited.insert(node);

    // Process node

    for (int neighbor : graph[node]) {
        dfs(neighbor, graph, visited);
    }
}

// Iterative DFS
void dfsIterative(int start, vector<vector<int>>& graph) {
    unordered_set<int> visited;
    stack<int> st;
    st.push(start);

    while (!st.empty()) {
        int node = st.top();
        st.pop();

        if (visited.count(node)) continue;
        visited.insert(node);

        // Process node

        for (int neighbor : graph[node]) {
            if (!visited.count(neighbor)) {
                st.push(neighbor);
            }
        }
    }
}
```

### BFS
```cpp
#include <vector>
#include <queue>
#include <unordered_set>
using namespace std;

void bfs(int start, vector<vector<int>>& graph) {
    unordered_set<int> visited;
    queue<int> q;
    q.push(start);
    visited.insert(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        // Process node

        for (int neighbor : graph[node]) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
}

// BFS with level tracking
int bfsWithLevel(int start, int end, vector<vector<int>>& graph) {
    unordered_set<int> visited;
    queue<int> q;
    q.push(start);
    visited.insert(start);
    int level = 0;

    while (!q.empty()) {
        int size = q.size();

        for (int i = 0; i < size; i++) {
            int node = q.front();
            q.pop();

            if (node == end) return level;

            for (int neighbor : graph[node]) {
                if (!visited.count(neighbor)) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }

        level++;
    }

    return -1;  // Not reachable
}
```

### Matrix DFS/BFS (4-directional)
```cpp
#include <vector>
#include <queue>
using namespace std;

// Direction vectors
int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

// DFS on matrix
void dfsMatrix(vector<vector<int>>& matrix, int i, int j,
               vector<vector<bool>>& visited) {
    int m = matrix.size(), n = matrix[0].size();

    if (i < 0 || i >= m || j < 0 || j >= n || visited[i][j]) {
        return;
    }

    visited[i][j] = true;
    // Process cell

    for (int d = 0; d < 4; d++) {
        dfsMatrix(matrix, i + dx[d], j + dy[d], visited);
    }
}

// BFS on matrix
void bfsMatrix(vector<vector<int>>& matrix, int startI, int startJ) {
    int m = matrix.size(), n = matrix[0].size();
    vector<vector<bool>> visited(m, vector<bool>(n, false));
    queue<pair<int, int>> q;

    q.push({startI, startJ});
    visited[startI][startJ] = true;

    while (!q.empty()) {
        auto [i, j] = q.front();
        q.pop();

        // Process cell

        for (int d = 0; d < 4; d++) {
            int ni = i + dx[d], nj = j + dy[d];
            if (ni >= 0 && ni < m && nj >= 0 && nj < n && !visited[ni][nj]) {
                visited[ni][nj] = true;
                q.push({ni, nj});
            }
        }
    }
}
```

### Topological Sort (Kahn's Algorithm - BFS)
```cpp
#include <vector>
#include <queue>
using namespace std;

vector<int> topologicalSort(int n, vector<vector<int>>& edges) {
    vector<vector<int>> graph(n);
    vector<int> inDegree(n, 0);

    // Build graph
    for (auto& edge : edges) {
        graph[edge[0]].push_back(edge[1]);
        inDegree[edge[1]]++;
    }

    // Find nodes with 0 in-degree
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (inDegree[i] == 0) q.push(i);
    }

    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : graph[node]) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    // Check for cycle
    if (result.size() != n) return {};  // Cycle exists

    return result;
}
```

### Topological Sort (DFS)
```cpp
#include <vector>
#include <stack>
using namespace std;

class Solution {
public:
    stack<int> order;
    vector<int> color;  // 0: white, 1: gray, 2: black
    bool hasCycle = false;

    void dfs(int node, vector<vector<int>>& graph) {
        color[node] = 1;  // Gray - visiting

        for (int neighbor : graph[node]) {
            if (color[neighbor] == 1) {
                hasCycle = true;
                return;
            }
            if (color[neighbor] == 0) {
                dfs(neighbor, graph);
            }
        }

        color[node] = 2;  // Black - done
        order.push(node);
    }

    vector<int> topologicalSort(int n, vector<vector<int>>& edges) {
        vector<vector<int>> graph(n);
        for (auto& edge : edges) {
            graph[edge[0]].push_back(edge[1]);
        }

        color.assign(n, 0);

        for (int i = 0; i < n; i++) {
            if (color[i] == 0) {
                dfs(i, graph);
            }
        }

        if (hasCycle) return {};

        vector<int> result;
        while (!order.empty()) {
            result.push_back(order.top());
            order.pop();
        }
        return result;
    }
};
```

### Dijkstra's Algorithm
```cpp
#include <vector>
#include <queue>
#include <climits>
using namespace std;

vector<int> dijkstra(int n, vector<vector<pair<int, int>>>& graph, int start) {
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    dist[start] = 0;
    pq.push({0, start});  // {distance, node}

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;  // Already processed with shorter distance

        for (auto& [v, weight] : graph[u]) {
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}
```

### Bellman-Ford Algorithm
```cpp
#include <vector>
#include <climits>
using namespace std;

vector<int> bellmanFord(int n, vector<vector<int>>& edges, int start) {
    vector<int> dist(n, INT_MAX);
    dist[start] = 0;

    // Relax all edges n-1 times
    for (int i = 0; i < n - 1; i++) {
        for (auto& edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }

    // Check for negative cycle
    for (auto& edge : edges) {
        int u = edge[0], v = edge[1], w = edge[2];
        if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
            return {};  // Negative cycle exists
        }
    }

    return dist;
}
```

### Floyd-Warshall Algorithm
```cpp
#include <vector>
#include <climits>
using namespace std;

vector<vector<int>> floydWarshall(int n, vector<vector<int>>& edges) {
    vector<vector<int>> dist(n, vector<int>(n, INT_MAX / 2));

    // Initialize
    for (int i = 0; i < n; i++) dist[i][i] = 0;
    for (auto& edge : edges) {
        dist[edge[0]][edge[1]] = edge[2];
    }

    // DP
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }

    return dist;
}
```

### Detect Cycle in Undirected Graph
```cpp
#include <vector>
using namespace std;

bool hasCycleUndirected(int node, int parent, vector<vector<int>>& graph,
                        vector<bool>& visited) {
    visited[node] = true;

    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            if (hasCycleUndirected(neighbor, node, graph, visited)) {
                return true;
            }
        } else if (neighbor != parent) {
            return true;
        }
    }

    return false;
}
```

### Number of Islands
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};

    void dfs(vector<vector<char>>& grid, int i, int j) {
        int m = grid.size(), n = grid[0].size();

        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == '0') {
            return;
        }

        grid[i][j] = '0';  // Mark as visited

        for (int d = 0; d < 4; d++) {
            dfs(grid, i + dx[d], j + dy[d]);
        }
    }

    int numIslands(vector<vector<char>>& grid) {
        int count = 0;
        int m = grid.size(), n = grid[0].size();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    dfs(grid, i, j);
                }
            }
        }

        return count;
    }
};
```

---

## Heap / Priority Queue

### Key Points
- Max heap: largest element at top
- Min heap: smallest element at top
- Insert: O(log n), Extract: O(log n), Peek: O(1)
- "Top K" problems often use heaps

### Usage in C++
```cpp
#include <queue>
#include <vector>
using namespace std;

// Min heap (default in C++ is max heap, so use greater<>)
priority_queue<int, vector<int>, greater<int>> minHeap;

// Max heap
priority_queue<int> maxHeap;

// Custom comparator for pairs (min heap by second element)
auto cmp = [](pair<int, int>& a, pair<int, int>& b) {
    return a.second > b.second;  // Greater = min heap
};
priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq(cmp);

// Operations
minHeap.push(5);
int top = minHeap.top();
minHeap.pop();
bool empty = minHeap.empty();
int size = minHeap.size();
```

### Top K Frequent Elements
```cpp
#include <vector>
#include <queue>
#include <unordered_map>
using namespace std;

vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int num : nums) freq[num]++;

    // Min heap of size k
    auto cmp = [](pair<int, int>& a, pair<int, int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq(cmp);

    for (auto& [num, count] : freq) {
        pq.push({num, count});
        if (pq.size() > k) pq.pop();
    }

    vector<int> result;
    while (!pq.empty()) {
        result.push_back(pq.top().first);
        pq.pop();
    }

    return result;
}
```

### Merge K Sorted Lists
```cpp
ListNode* mergeKLists(vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) {
        return a->val > b->val;
    };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);

    for (ListNode* list : lists) {
        if (list) pq.push(list);
    }

    ListNode dummy(0);
    ListNode* curr = &dummy;

    while (!pq.empty()) {
        ListNode* node = pq.top();
        pq.pop();

        curr->next = node;
        curr = curr->next;

        if (node->next) pq.push(node->next);
    }

    return dummy.next;
}
```

### Find Median from Data Stream
```cpp
#include <queue>
using namespace std;

class MedianFinder {
private:
    priority_queue<int> maxHeap;  // Lower half
    priority_queue<int, vector<int>, greater<int>> minHeap;  // Upper half

public:
    void addNum(int num) {
        maxHeap.push(num);
        minHeap.push(maxHeap.top());
        maxHeap.pop();

        if (minHeap.size() > maxHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }

    double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.top();
        }
        return (maxHeap.top() + minHeap.top()) / 2.0;
    }
};
```

---

## Trie

### Implementation
```cpp
#include <string>
#include <unordered_map>
using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    bool isEndOfWord = false;
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(const string& word) {
        TrieNode* curr = root;
        for (char c : word) {
            if (curr->children.find(c) == curr->children.end()) {
                curr->children[c] = new TrieNode();
            }
            curr = curr->children[c];
        }
        curr->isEndOfWord = true;
    }

    bool search(const string& word) {
        TrieNode* node = findNode(word);
        return node && node->isEndOfWord;
    }

    bool startsWith(const string& prefix) {
        return findNode(prefix) != nullptr;
    }

private:
    TrieNode* findNode(const string& s) {
        TrieNode* curr = root;
        for (char c : s) {
            if (curr->children.find(c) == curr->children.end()) {
                return nullptr;
            }
            curr = curr->children[c];
        }
        return curr;
    }
};
```

### Trie with Array (26 lowercase letters)
```cpp
class TrieNode {
public:
    TrieNode* children[26] = {nullptr};
    bool isEndOfWord = false;
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(const string& word) {
        TrieNode* curr = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!curr->children[idx]) {
                curr->children[idx] = new TrieNode();
            }
            curr = curr->children[idx];
        }
        curr->isEndOfWord = true;
    }

    bool search(const string& word) {
        TrieNode* node = findNode(word);
        return node && node->isEndOfWord;
    }

    bool startsWith(const string& prefix) {
        return findNode(prefix) != nullptr;
    }

private:
    TrieNode* findNode(const string& s) {
        TrieNode* curr = root;
        for (char c : s) {
            int idx = c - 'a';
            if (!curr->children[idx]) return nullptr;
            curr = curr->children[idx];
        }
        return curr;
    }
};
```

---

## Dynamic Programming

### Key Points
- Identify overlapping subproblems
- Define state and recurrence relation
- Choose top-down (memoization) or bottom-up (tabulation)
- Optimize space when only previous states needed

### Fibonacci (Classic Example)
```cpp
#include <vector>
using namespace std;

// Top-down (Memoization)
int fibMemo(int n, vector<int>& memo) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];
    return memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
}

// Bottom-up (Tabulation)
int fibTab(int n) {
    if (n <= 1) return n;
    vector<int> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

// Space optimized
int fibOptimized(int n) {
    if (n <= 1) return n;
    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

### Climbing Stairs
```cpp
int climbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

### House Robber
```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];

    int prev2 = nums[0];
    int prev1 = max(nums[0], nums[1]);

    for (int i = 2; i < n; i++) {
        int curr = max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}
```

### Coin Change
```cpp
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }

    return dp[amount] > amount ? -1 : dp[amount];
}
```

### Longest Common Subsequence (LCS)
```cpp
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i - 1] == text2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp[m][n];
}
```

### Longest Increasing Subsequence (LIS)
```cpp
#include <algorithm>
using namespace std;

// O(n^2) solution
int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);
    int maxLen = 1;

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
        maxLen = max(maxLen, dp[i]);
    }

    return maxLen;
}

// O(n log n) solution using binary search
int lengthOfLISOptimized(vector<int>& nums) {
    vector<int> tails;

    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }

    return tails.size();
}
```

### Edit Distance
```cpp
int minDistance(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + min({dp[i - 1][j],      // Delete
                                   dp[i][j - 1],       // Insert
                                   dp[i - 1][j - 1]}); // Replace
            }
        }
    }

    return dp[m][n];
}
```

### 0/1 Knapsack
```cpp
int knapsack(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = max(dp[i - 1][w],
                              dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    return dp[n][W];
}

// Space optimized (1D array)
int knapsackOptimized(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<int> dp(W + 1, 0);

    for (int i = 0; i < n; i++) {
        for (int w = W; w >= weights[i]; w--) {  // Reverse order!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }

    return dp[W];
}
```

### Unbounded Knapsack (Coin Change Variant)
```cpp
int unboundedKnapsack(vector<int>& weights, vector<int>& values, int W) {
    vector<int> dp(W + 1, 0);

    for (int w = 1; w <= W; w++) {
        for (int i = 0; i < weights.size(); i++) {
            if (weights[i] <= w) {
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
            }
        }
    }

    return dp[W];
}
```

### Word Break
```cpp
bool wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> dict(wordDict.begin(), wordDict.end());
    int n = s.length();
    vector<bool> dp(n + 1, false);
    dp[0] = true;

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < i; j++) {
            if (dp[j] && dict.count(s.substr(j, i - j))) {
                dp[i] = true;
                break;
            }
        }
    }

    return dp[n];
}
```

### Partition Equal Subset Sum
```cpp
bool canPartition(vector<int>& nums) {
    int total = accumulate(nums.begin(), nums.end(), 0);
    if (total % 2 != 0) return false;

    int target = total / 2;
    vector<bool> dp(target + 1, false);
    dp[0] = true;

    for (int num : nums) {
        for (int j = target; j >= num; j--) {
            dp[j] = dp[j] || dp[j - num];
        }
    }

    return dp[target];
}
```

### Unique Paths
```cpp
int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n, 1));

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }

    return dp[m - 1][n - 1];
}

// Space optimized
int uniquePathsOptimized(int m, int n) {
    vector<int> dp(n, 1);

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[j] += dp[j - 1];
        }
    }

    return dp[n - 1];
}
```

### Maximum Product Subarray
```cpp
int maxProduct(vector<int>& nums) {
    int maxProd = nums[0];
    int minProd = nums[0];
    int result = nums[0];

    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] < 0) swap(maxProd, minProd);

        maxProd = max(nums[i], maxProd * nums[i]);
        minProd = min(nums[i], minProd * nums[i]);

        result = max(result, maxProd);
    }

    return result;
}
```

---

## Intervals

### Key Points
- Sort by start time first
- Check overlap: `a.start < b.end && b.start < a.end`
- Merge: `[min(a.start, b.start), max(a.end, b.end)]`

### Merge Intervals
```cpp
#include <vector>
#include <algorithm>
using namespace std;

vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if (intervals.empty()) return {};

    sort(intervals.begin(), intervals.end());
    vector<vector<int>> result;
    result.push_back(intervals[0]);

    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i][0] <= result.back()[1]) {
            result.back()[1] = max(result.back()[1], intervals[i][1]);
        } else {
            result.push_back(intervals[i]);
        }
    }

    return result;
}
```

### Insert Interval
```cpp
vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
    vector<vector<int>> result;
    int i = 0, n = intervals.size();

    // Add all intervals before newInterval
    while (i < n && intervals[i][1] < newInterval[0]) {
        result.push_back(intervals[i++]);
    }

    // Merge overlapping intervals
    while (i < n && intervals[i][0] <= newInterval[1]) {
        newInterval[0] = min(newInterval[0], intervals[i][0]);
        newInterval[1] = max(newInterval[1], intervals[i][1]);
        i++;
    }
    result.push_back(newInterval);

    // Add remaining intervals
    while (i < n) {
        result.push_back(intervals[i++]);
    }

    return result;
}
```

### Non-overlapping Intervals (Minimum removals)
```cpp
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    if (intervals.empty()) return 0;

    // Sort by end time (greedy)
    sort(intervals.begin(), intervals.end(), [](auto& a, auto& b) {
        return a[1] < b[1];
    });

    int count = 0;
    int end = intervals[0][1];

    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i][0] < end) {
            count++;  // Remove this interval
        } else {
            end = intervals[i][1];
        }
    }

    return count;
}
```

### Meeting Rooms II (Minimum rooms needed)
```cpp
int minMeetingRooms(vector<vector<int>>& intervals) {
    vector<int> starts, ends;

    for (auto& interval : intervals) {
        starts.push_back(interval[0]);
        ends.push_back(interval[1]);
    }

    sort(starts.begin(), starts.end());
    sort(ends.begin(), ends.end());

    int rooms = 0, endPtr = 0;

    for (int i = 0; i < starts.size(); i++) {
        if (starts[i] < ends[endPtr]) {
            rooms++;
        } else {
            endPtr++;
        }
    }

    return rooms;
}
```

---

## Geometry

### Distance Between Two Points
```cpp
#include <cmath>

double distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}
```

### Overlapping Rectangles
```cpp
bool doRectanglesOverlap(int x1, int y1, int x2, int y2,  // rect1: bottom-left, top-right
                         int x3, int y3, int x4, int y4) { // rect2: bottom-left, top-right
    // No overlap if one is to the left or above the other
    if (x1 >= x4 || x3 >= x2) return false;  // One is to the left
    if (y1 >= y4 || y3 >= y2) return false;  // One is below
    return true;
}
```

### Overlapping Circles
```cpp
bool doCirclesOverlap(double x1, double y1, double r1,
                      double x2, double y2, double r2) {
    double dist = distance(x1, y1, x2, y2);
    return dist <= r1 + r2;
}
```

---

## Bit Manipulation

### Key Operations
```cpp
// Check if bit is set
bool isSet = (n & (1 << i)) != 0;

// Set bit
n = n | (1 << i);

// Clear bit
n = n & ~(1 << i);

// Toggle bit
n = n ^ (1 << i);

// Count set bits (Brian Kernighan)
int countBits(int n) {
    int count = 0;
    while (n) {
        n &= (n - 1);  // Removes rightmost set bit
        count++;
    }
    return count;
}

// Check if power of 2
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Get rightmost set bit
int rightmostBit = n & (-n);

// XOR properties:
// a ^ a = 0
// a ^ 0 = a
// a ^ b ^ a = b
```

### Single Number (All appear twice except one)
```cpp
int singleNumber(vector<int>& nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;
    }
    return result;
}
```

### Single Number II (All appear thrice except one)
```cpp
int singleNumberII(vector<int>& nums) {
    int ones = 0, twos = 0;
    for (int num : nums) {
        ones = (ones ^ num) & ~twos;
        twos = (twos ^ num) & ~ones;
    }
    return ones;
}
```

### Missing Number
```cpp
int missingNumber(vector<int>& nums) {
    int n = nums.size();
    int result = n;  // Start with n
    for (int i = 0; i < n; i++) {
        result ^= i ^ nums[i];
    }
    return result;
}
```

### Reverse Bits
```cpp
uint32_t reverseBits(uint32_t n) {
    uint32_t result = 0;
    for (int i = 0; i < 32; i++) {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    return result;
}
```

---

## Disjoint Set Union (DSU)

### Implementation
```cpp
#include <vector>
using namespace std;

class DSU {
private:
    vector<int> parent, rank;

public:
    DSU(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;  // Already in same set

        // Union by rank
        if (rank[px] < rank[py]) swap(px, py);
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;

        return true;
    }

    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};
```

### Number of Connected Components
```cpp
int countComponents(int n, vector<vector<int>>& edges) {
    DSU dsu(n);
    int components = n;

    for (auto& edge : edges) {
        if (dsu.unite(edge[0], edge[1])) {
            components--;
        }
    }

    return components;
}
```

### Smallest String with Swaps
```cpp
string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
    int n = s.size();
    DSU dsu(n);

    for (auto& pair : pairs) {
        dsu.unite(pair[0], pair[1]);
    }

    // Group indices by their root
    unordered_map<int, vector<int>> groups;
    for (int i = 0; i < n; i++) {
        groups[dsu.find(i)].push_back(i);
    }

    // Sort characters within each group
    string result = s;
    for (auto& [root, indices] : groups) {
        string chars;
        for (int idx : indices) {
            chars += s[idx];
        }
        sort(chars.begin(), chars.end());

        for (int i = 0; i < indices.size(); i++) {
            result[indices[i]] = chars[i];
        }
    }

    return result;
}
```

---

## Monotonic Stack

### Key Points
- Maintains increasing or decreasing order
- Useful for "next greater/smaller element" problems

### Next Greater Element
```cpp
vector<int> nextGreaterElement(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st;  // Stack of indices

    for (int i = 0; i < n; i++) {
        while (!st.empty() && nums[i] > nums[st.top()]) {
            result[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }

    return result;
}
```

### Daily Temperatures
```cpp
vector<int> dailyTemperatures(vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> result(n, 0);
    stack<int> st;

    for (int i = 0; i < n; i++) {
        while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
            int prevIdx = st.top();
            st.pop();
            result[prevIdx] = i - prevIdx;
        }
        st.push(i);
    }

    return result;
}
```

### Largest Rectangle in Histogram
```cpp
int largestRectangleArea(vector<int>& heights) {
    int n = heights.size();
    stack<int> st;
    int maxArea = 0;

    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];

        while (!st.empty() && h < heights[st.top()]) {
            int height = heights[st.top()];
            st.pop();
            int width = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, height * width);
        }

        st.push(i);
    }

    return maxArea;
}
```

### Trapping Rain Water
```cpp
int trap(vector<int>& height) {
    int n = height.size();
    if (n == 0) return 0;

    // Two pointer approach
    int left = 0, right = n - 1;
    int leftMax = 0, rightMax = 0;
    int water = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) {
                leftMax = height[left];
            } else {
                water += leftMax - height[left];
            }
            left++;
        } else {
            if (height[right] >= rightMax) {
                rightMax = height[right];
            } else {
                water += rightMax - height[right];
            }
            right--;
        }
    }

    return water;
}
```

---

## Notable Algorithms

### Robot Room Cleaner
```cpp
class Solution {
private:
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1};
    set<pair<int, int>> visited;
    Robot* robot;

    void goBack() {
        robot->turnRight();
        robot->turnRight();
        robot->move();
        robot->turnRight();
        robot->turnRight();
    }

    void backtrack(int row, int col, int d) {
        visited.insert({row, col});
        robot->clean();

        for (int i = 0; i < 4; i++) {
            int newD = (d + i) % 4;
            int newRow = row + dx[newD];
            int newCol = col + dy[newD];

            if (visited.find({newRow, newCol}) == visited.end() && robot->move()) {
                backtrack(newRow, newCol, newD);
                goBack();
            }

            robot->turnRight();
        }
    }

public:
    void cleanRoom(Robot& r) {
        robot = &r;
        backtrack(0, 0, 0);
    }
};
```

### Peak Valley (Stock Trading)
```cpp
int maxProfit(vector<int>& prices) {
    int maxProfit = 0;
    int i = 0;
    int n = prices.size();

    while (i < n - 1) {
        // Find valley
        while (i < n - 1 && prices[i] >= prices[i + 1]) {
            i++;
        }
        int valley = prices[i];

        // Find peak
        while (i < n - 1 && prices[i] <= prices[i + 1]) {
            i++;
        }
        int peak = prices[i];

        maxProfit += peak - valley;
    }

    return maxProfit;
}

// Simpler approach: collect all upward slopes
int maxProfitSimple(vector<int>& prices) {
    int profit = 0;
    for (int i = 1; i < prices.size(); i++) {
        if (prices[i] > prices[i - 1]) {
            profit += prices[i] - prices[i - 1];
        }
    }
    return profit;
}
```

### Clone Graph
```cpp
class Solution {
    unordered_map<Node*, Node*> visited;

public:
    Node* cloneGraph(Node* node) {
        if (!node) return nullptr;

        if (visited.count(node)) {
            return visited[node];
        }

        Node* clone = new Node(node->val);
        visited[node] = clone;

        for (Node* neighbor : node->neighbors) {
            clone->neighbors.push_back(cloneGraph(neighbor));
        }

        return clone;
    }
};
```

### Course Schedule (Cycle Detection)
```cpp
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses);
    vector<int> inDegree(numCourses, 0);

    for (auto& pre : prerequisites) {
        graph[pre[1]].push_back(pre[0]);
        inDegree[pre[0]]++;
    }

    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (inDegree[i] == 0) q.push(i);
    }

    int count = 0;
    while (!q.empty()) {
        int course = q.front();
        q.pop();
        count++;

        for (int next : graph[course]) {
            if (--inDegree[next] == 0) {
                q.push(next);
            }
        }
    }

    return count == numCourses;
}
```

### Alien Dictionary (Topological Sort)
```cpp
string alienOrder(vector<string>& words) {
    unordered_map<char, unordered_set<char>> graph;
    unordered_map<char, int> inDegree;

    // Initialize all characters
    for (const string& word : words) {
        for (char c : word) {
            inDegree[c] = 0;
        }
    }

    // Build graph
    for (int i = 0; i < words.size() - 1; i++) {
        string& w1 = words[i];
        string& w2 = words[i + 1];

        // Check for invalid case: "abc" before "ab"
        if (w1.length() > w2.length() && w1.substr(0, w2.length()) == w2) {
            return "";
        }

        for (int j = 0; j < min(w1.length(), w2.length()); j++) {
            if (w1[j] != w2[j]) {
                if (!graph[w1[j]].count(w2[j])) {
                    graph[w1[j]].insert(w2[j]);
                    inDegree[w2[j]]++;
                }
                break;
            }
        }
    }

    // Topological sort
    queue<char> q;
    for (auto& [c, deg] : inDegree) {
        if (deg == 0) q.push(c);
    }

    string result;
    while (!q.empty()) {
        char c = q.front();
        q.pop();
        result += c;

        for (char next : graph[c]) {
            if (--inDegree[next] == 0) {
                q.push(next);
            }
        }
    }

    return result.length() == inDegree.size() ? result : "";
}
```

---

## Quick Reference: Time Complexities

| Data Structure | Access | Search | Insert | Delete |
|---------------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) |
| Stack | O(n) | O(n) | O(1) | O(1) |
| Queue | O(n) | O(n) | O(1) | O(1) |
| Hash Table | N/A | O(1)* | O(1)* | O(1)* |
| BST | O(log n)* | O(log n)* | O(log n)* | O(log n)* |
| Heap | O(1) | O(n) | O(log n) | O(log n) |
| Trie | N/A | O(m) | O(m) | O(m) |

*Average case, worst case may differ

| Algorithm | Time | Space |
|-----------|------|-------|
| Binary Search | O(log n) | O(1) |
| DFS/BFS | O(V + E) | O(V) |
| Dijkstra | O((V+E) log V) | O(V) |
| Bellman-Ford | O(VE) | O(V) |
| Floyd-Warshall | O(V³) | O(V²) |
| Quick Sort | O(n log n)* | O(log n) |
| Merge Sort | O(n log n) | O(n) |
| Heap Sort | O(n log n) | O(1) |

---

## Common Patterns Summary

1. **Sliding Window**: Subarray/substring problems with contiguous elements
2. **Two Pointers**: Sorted arrays, pair finding, partitioning
3. **Fast & Slow Pointers**: Cycle detection, middle finding
4. **Merge Intervals**: Overlapping intervals
5. **Cyclic Sort**: Missing/duplicate numbers in range [1, n]
6. **In-place Reversal**: Linked list operations
7. **BFS**: Level-order traversal, shortest path (unweighted)
8. **DFS**: Path finding, tree traversals, backtracking
9. **Two Heaps**: Median finding, scheduling
10. **Subsets/Permutations**: Backtracking with recursion
11. **Binary Search**: Sorted arrays, search space reduction
12. **Top K Elements**: Heap-based selection
13. **K-way Merge**: Multiple sorted lists
14. **Topological Sort**: Dependency ordering
15. **0/1 Knapsack**: Subset selection with constraints
16. **Unbounded Knapsack**: Infinite supply selection
17. **Fibonacci Numbers**: Sequence patterns
18. **Palindromic Subsequence**: String DP
19. **Longest Common Subsequence**: Two-string comparison
20. **Monotonic Stack**: Next greater/smaller element

---

## LeetCode Problem Mapping by Pattern

### Array Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 53 | Maximum Subarray | Kadane's Algorithm | Medium |
| 121 | Best Time to Buy and Sell Stock | Kadane's variant | Easy |
| 152 | Maximum Product Subarray | DP with min/max tracking | Medium |
| 238 | Product of Array Except Self | Prefix/Suffix products | Medium |
| 41 | First Missing Positive | Index as hash key | Hard |
| 287 | Find the Duplicate Number | Floyd's cycle detection | Medium |
| 442 | Find All Duplicates in Array | Index as hash key | Medium |
| 448 | Find All Numbers Disappeared | Index as hash key | Easy |
| 75 | Sort Colors | Dutch National Flag | Medium |
| 31 | Next Permutation | Two pointers + reverse | Medium |
| 189 | Rotate Array | Reverse trick | Medium |
| 169 | Majority Element | Boyer-Moore voting | Easy |
| 229 | Majority Element II | Boyer-Moore extended | Medium |

### Two Pointers Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 1 | Two Sum | Hash map | Easy |
| 167 | Two Sum II - Sorted | Two pointers opposite | Medium |
| 15 | 3Sum | Sort + Two pointers | Medium |
| 16 | 3Sum Closest | Sort + Two pointers | Medium |
| 18 | 4Sum | Sort + Two pointers | Medium |
| 11 | Container With Most Water | Two pointers greedy | Medium |
| 42 | Trapping Rain Water | Two pointers / Stack | Hard |
| 125 | Valid Palindrome | Two pointers | Easy |
| 680 | Valid Palindrome II | Two pointers + greedy | Easy |
| 26 | Remove Duplicates from Sorted Array | Fast/slow pointers | Easy |
| 27 | Remove Element | Fast/slow pointers | Easy |
| 283 | Move Zeroes | Fast/slow pointers | Easy |
| 344 | Reverse String | Two pointers swap | Easy |
| 88 | Merge Sorted Array | Two pointers from end | Easy |

### Sliding Window Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 3 | Longest Substring Without Repeating | Variable window + Set | Medium |
| 76 | Minimum Window Substring | Variable window + Map | Hard |
| 209 | Minimum Size Subarray Sum | Variable window | Medium |
| 424 | Longest Repeating Character Replacement | Variable window | Medium |
| 567 | Permutation in String | Fixed window + freq | Medium |
| 438 | Find All Anagrams in String | Fixed window + freq | Medium |
| 239 | Sliding Window Maximum | Monotonic deque | Hard |
| 480 | Sliding Window Median | Two heaps | Hard |
| 904 | Fruit Into Baskets | Variable window | Medium |
| 1004 | Max Consecutive Ones III | Variable window | Medium |
| 1438 | Longest Subarray with Limit | Monotonic deque | Medium |

### Binary Search Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 704 | Binary Search | Standard | Easy |
| 33 | Search in Rotated Sorted Array | Modified BS | Medium |
| 81 | Search in Rotated Sorted Array II | Modified BS + duplicates | Medium |
| 153 | Find Minimum in Rotated Sorted Array | Modified BS | Medium |
| 154 | Find Minimum in Rotated Sorted Array II | Modified BS + duplicates | Hard |
| 34 | Find First and Last Position | Lower/Upper bound | Medium |
| 35 | Search Insert Position | Lower bound | Easy |
| 74 | Search a 2D Matrix | 2D to 1D BS | Medium |
| 240 | Search a 2D Matrix II | BS per row/col | Medium |
| 162 | Find Peak Element | BS on answer | Medium |
| 852 | Peak Index in Mountain Array | BS on answer | Medium |
| 875 | Koko Eating Bananas | BS on answer | Medium |
| 1011 | Capacity To Ship Packages | BS on answer | Medium |
| 410 | Split Array Largest Sum | BS on answer | Hard |
| 69 | Sqrt(x) | BS on answer | Easy |
| 278 | First Bad Version | BS | Easy |

### Linked List Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 206 | Reverse Linked List | Iterative/Recursive | Easy |
| 92 | Reverse Linked List II | Partial reverse | Medium |
| 25 | Reverse Nodes in k-Group | k-group reverse | Hard |
| 141 | Linked List Cycle | Fast/slow pointers | Easy |
| 142 | Linked List Cycle II | Floyd's algorithm | Medium |
| 876 | Middle of the Linked List | Fast/slow pointers | Easy |
| 19 | Remove Nth Node From End | Two pointers k-apart | Medium |
| 21 | Merge Two Sorted Lists | Dummy node | Easy |
| 23 | Merge k Sorted Lists | Heap / Divide conquer | Hard |
| 143 | Reorder List | Find mid + reverse + merge | Medium |
| 234 | Palindrome Linked List | Fast/slow + reverse | Easy |
| 160 | Intersection of Two Linked Lists | Two pointers | Easy |
| 328 | Odd Even Linked List | Two pointers | Medium |
| 138 | Copy List with Random Pointer | Hash map / Interweaving | Medium |
| 146 | LRU Cache | Hash map + DLL | Medium |
| 460 | LFU Cache | Hash maps + DLL | Hard |

### Stack Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 20 | Valid Parentheses | Stack matching | Easy |
| 155 | Min Stack | Auxiliary stack | Medium |
| 232 | Implement Queue using Stacks | Two stacks | Easy |
| 225 | Implement Stack using Queues | Two queues | Easy |
| 150 | Evaluate Reverse Polish Notation | Stack | Medium |
| 71 | Simplify Path | Stack | Medium |
| 394 | Decode String | Stack | Medium |
| 735 | Asteroid Collision | Stack simulation | Medium |
| 856 | Score of Parentheses | Stack | Medium |
| 1249 | Minimum Remove to Make Valid | Stack + Set | Medium |

### Monotonic Stack Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 496 | Next Greater Element I | Monotonic decreasing | Easy |
| 503 | Next Greater Element II | Circular + monotonic | Medium |
| 739 | Daily Temperatures | Monotonic decreasing | Medium |
| 84 | Largest Rectangle in Histogram | Monotonic increasing | Hard |
| 85 | Maximal Rectangle | Histogram per row | Hard |
| 42 | Trapping Rain Water | Monotonic / Two pointers | Hard |
| 901 | Online Stock Span | Monotonic decreasing | Medium |
| 907 | Sum of Subarray Minimums | Monotonic + contribution | Medium |
| 1019 | Next Greater Node In Linked List | Monotonic stack | Medium |
| 402 | Remove K Digits | Monotonic increasing | Medium |

### Tree Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 94 | Binary Tree Inorder Traversal | DFS iterative/recursive | Easy |
| 144 | Binary Tree Preorder Traversal | DFS | Easy |
| 145 | Binary Tree Postorder Traversal | DFS | Easy |
| 102 | Binary Tree Level Order Traversal | BFS | Medium |
| 107 | Binary Tree Level Order II | BFS + reverse | Medium |
| 103 | Binary Tree Zigzag Level Order | BFS + flag | Medium |
| 104 | Maximum Depth of Binary Tree | DFS | Easy |
| 111 | Minimum Depth of Binary Tree | BFS/DFS | Easy |
| 110 | Balanced Binary Tree | DFS with height | Easy |
| 226 | Invert Binary Tree | DFS/BFS | Easy |
| 101 | Symmetric Tree | DFS compare | Easy |
| 100 | Same Tree | DFS compare | Easy |
| 572 | Subtree of Another Tree | DFS + same tree | Easy |
| 112 | Path Sum | DFS | Easy |
| 113 | Path Sum II | DFS + backtrack | Medium |
| 437 | Path Sum III | Prefix sum + DFS | Medium |
| 124 | Binary Tree Maximum Path Sum | DFS post-order | Hard |
| 543 | Diameter of Binary Tree | DFS with global max | Easy |
| 236 | Lowest Common Ancestor | DFS | Medium |
| 235 | LCA of BST | BST property | Medium |
| 98 | Validate Binary Search Tree | DFS with range | Medium |
| 230 | Kth Smallest Element in BST | Inorder + count | Medium |
| 450 | Delete Node in BST | BST operations | Medium |
| 701 | Insert into BST | BST operations | Medium |
| 297 | Serialize and Deserialize | BFS/DFS | Hard |
| 105 | Construct from Preorder and Inorder | Divide & conquer | Medium |
| 106 | Construct from Inorder and Postorder | Divide & conquer | Medium |
| 114 | Flatten Binary Tree to Linked List | Morris / Stack | Medium |
| 199 | Binary Tree Right Side View | BFS / DFS | Medium |
| 116 | Populating Next Right Pointers | BFS / O(1) space | Medium |
| 222 | Count Complete Tree Nodes | Binary search | Medium |
| 958 | Check Completeness | BFS | Medium |

### Graph Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 200 | Number of Islands | DFS/BFS flood fill | Medium |
| 695 | Max Area of Island | DFS/BFS | Medium |
| 733 | Flood Fill | DFS/BFS | Easy |
| 463 | Island Perimeter | Counting | Easy |
| 547 | Number of Provinces | DFS/BFS/Union Find | Medium |
| 130 | Surrounded Regions | DFS from boundary | Medium |
| 417 | Pacific Atlantic Water Flow | Multi-source DFS | Medium |
| 994 | Rotting Oranges | Multi-source BFS | Medium |
| 286 | Walls and Gates | Multi-source BFS | Medium |
| 127 | Word Ladder | BFS shortest path | Hard |
| 126 | Word Ladder II | BFS + DFS backtrack | Hard |
| 752 | Open the Lock | BFS | Medium |
| 133 | Clone Graph | DFS/BFS + hash map | Medium |
| 207 | Course Schedule | Topological sort (cycle) | Medium |
| 210 | Course Schedule II | Topological sort (order) | Medium |
| 269 | Alien Dictionary | Topological sort | Hard |
| 310 | Minimum Height Trees | Topological (leaf removal) | Medium |
| 802 | Find Eventual Safe States | Reverse topo / DFS | Medium |
| 743 | Network Delay Time | Dijkstra | Medium |
| 787 | Cheapest Flights Within K Stops | Bellman-Ford / BFS | Medium |
| 1091 | Shortest Path in Binary Matrix | BFS | Medium |
| 1584 | Min Cost to Connect All Points | Prim's / Kruskal's MST | Medium |
| 684 | Redundant Connection | Union Find | Medium |
| 685 | Redundant Connection II | Union Find (directed) | Hard |
| 323 | Number of Connected Components | Union Find / DFS | Medium |
| 261 | Graph Valid Tree | Union Find / DFS | Medium |
| 990 | Satisfiability of Equality | Union Find | Medium |
| 721 | Accounts Merge | Union Find + sort | Medium |
| 1202 | Smallest String With Swaps | Union Find + sort | Medium |
| 785 | Is Graph Bipartite | BFS/DFS coloring | Medium |
| 886 | Possible Bipartition | BFS/DFS coloring | Medium |
| 329 | Longest Increasing Path in Matrix | DFS + memo | Hard |
| 79 | Word Search | DFS backtrack | Medium |
| 212 | Word Search II | Trie + DFS | Hard |
| 489 | Robot Room Cleaner | DFS + backtrack | Hard |

### Heap/Priority Queue Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 215 | Kth Largest Element in Array | Quick select / Heap | Medium |
| 347 | Top K Frequent Elements | Min heap of size K | Medium |
| 692 | Top K Frequent Words | Min heap + comparator | Medium |
| 703 | Kth Largest Element in Stream | Min heap | Easy |
| 295 | Find Median from Data Stream | Two heaps | Hard |
| 480 | Sliding Window Median | Two heaps + lazy delete | Hard |
| 23 | Merge k Sorted Lists | Min heap | Hard |
| 373 | Find K Pairs with Smallest Sums | Min heap | Medium |
| 378 | Kth Smallest Element in Sorted Matrix | Min heap / BS | Medium |
| 767 | Reorganize String | Max heap greedy | Medium |
| 621 | Task Scheduler | Max heap / counting | Medium |
| 358 | Rearrange String k Distance Apart | Max heap | Hard |
| 1046 | Last Stone Weight | Max heap | Easy |
| 973 | K Closest Points to Origin | Min/Max heap | Medium |
| 1167 | Minimum Cost to Connect Sticks | Min heap | Medium |
| 253 | Meeting Rooms II | Min heap / sweep line | Medium |
| 630 | Course Schedule III | Max heap greedy | Hard |

### Dynamic Programming Problems

#### 1D DP (Linear)
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 70 | Climbing Stairs | Fibonacci | Easy |
| 509 | Fibonacci Number | Fibonacci | Easy |
| 746 | Min Cost Climbing Stairs | Fibonacci variant | Easy |
| 198 | House Robber | Linear DP | Medium |
| 213 | House Robber II | Circular DP | Medium |
| 337 | House Robber III | Tree DP | Medium |
| 53 | Maximum Subarray | Kadane's | Medium |
| 152 | Maximum Product Subarray | Track min/max | Medium |
| 300 | Longest Increasing Subsequence | LIS | Medium |
| 673 | Number of Longest Increasing Subsequence | LIS + count | Medium |
| 1143 | Longest Common Subsequence | 2D DP | Medium |
| 583 | Delete Operation for Two Strings | LCS variant | Medium |
| 72 | Edit Distance | 2D DP | Medium |
| 115 | Distinct Subsequences | 2D DP | Hard |
| 97 | Interleaving String | 2D DP | Medium |

#### Knapsack Pattern
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 416 | Partition Equal Subset Sum | 0/1 Knapsack | Medium |
| 494 | Target Sum | 0/1 Knapsack / Count | Medium |
| 474 | Ones and Zeroes | 2D Knapsack | Medium |
| 1049 | Last Stone Weight II | 0/1 Knapsack | Medium |
| 322 | Coin Change | Unbounded Knapsack | Medium |
| 518 | Coin Change II | Unbounded (count ways) | Medium |
| 377 | Combination Sum IV | Unbounded (permutation) | Medium |
| 279 | Perfect Squares | Unbounded Knapsack | Medium |

#### String DP
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 5 | Longest Palindromic Substring | Expand center / DP | Medium |
| 647 | Palindromic Substrings | Expand center / DP | Medium |
| 516 | Longest Palindromic Subsequence | 2D DP | Medium |
| 1312 | Min Insertions for Palindrome | LPS variant | Hard |
| 131 | Palindrome Partitioning | Backtrack + DP | Medium |
| 132 | Palindrome Partitioning II | DP | Hard |
| 139 | Word Break | DP + hash set | Medium |
| 140 | Word Break II | DP + backtrack | Hard |
| 10 | Regular Expression Matching | 2D DP | Hard |
| 44 | Wildcard Matching | 2D DP | Hard |
| 1035 | Uncrossed Lines | LCS | Medium |
| 712 | Minimum ASCII Delete Sum | LCS variant | Medium |

#### Grid DP
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 62 | Unique Paths | Grid DP | Medium |
| 63 | Unique Paths II | Grid DP + obstacles | Medium |
| 64 | Minimum Path Sum | Grid DP | Medium |
| 120 | Triangle | Grid DP | Medium |
| 931 | Minimum Falling Path Sum | Grid DP | Medium |
| 221 | Maximal Square | Grid DP | Medium |
| 85 | Maximal Rectangle | Histogram + DP | Hard |
| 174 | Dungeon Game | Reverse grid DP | Hard |
| 741 | Cherry Pickup | 3D DP (two paths) | Hard |
| 1463 | Cherry Pickup II | 3D DP | Hard |

#### Interval DP
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 516 | Longest Palindromic Subsequence | Interval DP | Medium |
| 1000 | Minimum Cost to Merge Stones | Interval DP | Hard |
| 312 | Burst Balloons | Interval DP | Hard |
| 1547 | Minimum Cost to Cut a Stick | Interval DP | Hard |
| 87 | Scramble String | Interval DP + memo | Hard |

#### State Machine DP
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 121 | Best Time to Buy and Sell Stock | Single transaction | Easy |
| 122 | Best Time to Buy and Sell Stock II | Unlimited | Medium |
| 123 | Best Time to Buy and Sell Stock III | 2 transactions | Hard |
| 188 | Best Time to Buy and Sell Stock IV | K transactions | Hard |
| 309 | Best Time with Cooldown | State machine | Medium |
| 714 | Best Time with Transaction Fee | State machine | Medium |

#### Other DP Patterns
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 91 | Decode Ways | Linear DP | Medium |
| 639 | Decode Ways II | Linear DP + wildcard | Hard |
| 96 | Unique Binary Search Trees | Catalan numbers | Medium |
| 95 | Unique Binary Search Trees II | Catalan + construct | Medium |
| 343 | Integer Break | Math / DP | Medium |
| 368 | Largest Divisible Subset | LIS variant | Medium |
| 1048 | Longest String Chain | Hash + LIS | Medium |
| 1027 | Longest Arithmetic Subsequence | 2D DP | Medium |
| 264 | Ugly Number II | Multi-pointer DP | Medium |
| 313 | Super Ugly Number | Heap / DP | Medium |
| 354 | Russian Doll Envelopes | Sort + LIS | Hard |
| 403 | Frog Jump | DP + hash set | Hard |
| 818 | Race Car | BFS / DP | Hard |
| 1478 | Allocate Mailboxes | Interval + memo | Hard |

### Backtracking Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 46 | Permutations | Backtrack | Medium |
| 47 | Permutations II | Backtrack + skip dups | Medium |
| 78 | Subsets | Backtrack | Medium |
| 90 | Subsets II | Backtrack + skip dups | Medium |
| 39 | Combination Sum | Backtrack unlimited | Medium |
| 40 | Combination Sum II | Backtrack once each | Medium |
| 216 | Combination Sum III | Backtrack k numbers | Medium |
| 77 | Combinations | Backtrack | Medium |
| 17 | Letter Combinations of Phone | Backtrack | Medium |
| 22 | Generate Parentheses | Backtrack | Medium |
| 51 | N-Queens | Backtrack | Hard |
| 52 | N-Queens II | Backtrack count | Hard |
| 37 | Sudoku Solver | Backtrack | Hard |
| 79 | Word Search | DFS backtrack | Medium |
| 212 | Word Search II | Trie + backtrack | Hard |
| 93 | Restore IP Addresses | Backtrack | Medium |
| 131 | Palindrome Partitioning | Backtrack | Medium |
| 698 | Partition to K Equal Sum Subsets | Backtrack + prune | Medium |

### Interval Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 56 | Merge Intervals | Sort + merge | Medium |
| 57 | Insert Interval | Binary search / linear | Medium |
| 435 | Non-overlapping Intervals | Greedy (sort by end) | Medium |
| 252 | Meeting Rooms | Sort + check | Easy |
| 253 | Meeting Rooms II | Min heap / sweep | Medium |
| 986 | Interval List Intersections | Two pointers | Medium |
| 759 | Employee Free Time | Merge + heap | Hard |
| 1288 | Remove Covered Intervals | Sort + greedy | Medium |

### Trie Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 208 | Implement Trie | Basic Trie | Medium |
| 211 | Design Add and Search Words | Trie + DFS | Medium |
| 212 | Word Search II | Trie + backtrack | Hard |
| 648 | Replace Words | Trie prefix | Medium |
| 677 | Map Sum Pairs | Trie + value | Medium |
| 720 | Longest Word in Dictionary | Trie + BFS/DFS | Medium |
| 421 | Maximum XOR of Two Numbers | Bitwise Trie | Medium |
| 1268 | Search Suggestions System | Trie + DFS | Medium |

### Bit Manipulation Problems
| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 136 | Single Number | XOR | Easy |
| 137 | Single Number II | Bit counting | Medium |
| 260 | Single Number III | XOR + bit split | Medium |
| 268 | Missing Number | XOR / Math | Easy |
| 190 | Reverse Bits | Bit operations | Easy |
| 191 | Number of 1 Bits | Brian Kernighan | Easy |
| 338 | Counting Bits | DP + bit | Easy |
| 371 | Sum of Two Integers | Bit manipulation | Medium |
| 201 | Bitwise AND of Range | Bit shift | Medium |
| 318 | Maximum Product of Word Lengths | Bit mask | Medium |

---

## Solution Descriptions for Key Algorithms

### Array Solutions

**Kadane's Algorithm (LC #53 Maximum Subarray)**
- **Problem**: Find contiguous subarray with largest sum
- **Approach**: Track current sum and global max. At each element, decide to extend current subarray or start fresh
- **Key Insight**: `maxCurrent = max(arr[i], maxCurrent + arr[i])` - if adding current element makes sum worse than starting fresh, restart
- **Time**: O(n), **Space**: O(1)

**Dutch National Flag (LC #75 Sort Colors)**
- **Problem**: Sort array containing only 0, 1, 2 in-place
- **Approach**: Three pointers - low (0s boundary), mid (current), high (2s boundary)
- **Key Insight**: Elements before low are 0s, after high are 2s, between low and mid are 1s
- **Time**: O(n), **Space**: O(1)

**Next Permutation (LC #31)**
- **Problem**: Rearrange to next lexicographically greater permutation
- **Approach**: (1) Find first decreasing from right (2) Find just larger element (3) Swap (4) Reverse suffix
- **Key Insight**: The suffix after the swap point is in descending order, reversing gives smallest suffix
- **Time**: O(n), **Space**: O(1)

### Two Pointers Solutions

**3Sum (LC #15)**
- **Problem**: Find all triplets that sum to zero
- **Approach**: Sort array, fix first element, use two pointers for remaining two
- **Key Insight**: Skip duplicates to avoid repeated triplets
- **Time**: O(n²), **Space**: O(1) excluding output

**Container With Most Water (LC #11)**
- **Problem**: Find two lines that hold most water
- **Approach**: Two pointers at ends, move the shorter one inward
- **Key Insight**: Moving shorter line might find taller one; moving taller never improves area
- **Time**: O(n), **Space**: O(1)

**Trapping Rain Water (LC #42)**
- **Problem**: Calculate trapped water after rain
- **Approach**: Two pointers with left_max and right_max tracking
- **Key Insight**: Water at position = min(left_max, right_max) - height[i]
- **Time**: O(n), **Space**: O(1)

### Sliding Window Solutions

**Longest Substring Without Repeating (LC #3)**
- **Problem**: Find longest substring with unique characters
- **Approach**: Expand right, shrink left when duplicate found using set/map
- **Key Insight**: Maintain window invariant: no duplicates in current window
- **Time**: O(n), **Space**: O(min(n, alphabet_size))

**Minimum Window Substring (LC #76)**
- **Problem**: Find minimum window containing all chars of t
- **Approach**: Expand to satisfy, contract to minimize while valid
- **Key Insight**: Use "formed" counter to track when window is valid
- **Time**: O(s + t), **Space**: O(s + t)

### Binary Search Solutions

**Search in Rotated Sorted Array (LC #33)**
- **Problem**: Search in rotated sorted array
- **Approach**: Determine which half is sorted, decide which half to search
- **Key Insight**: One half is always sorted; check if target is in sorted half
- **Time**: O(log n), **Space**: O(1)

**Binary Search on Answer (LC #875 Koko Eating Bananas)**
- **Problem**: Find minimum eating speed to finish in H hours
- **Approach**: Binary search on speed from 1 to max(piles)
- **Key Insight**: Monotonic property - if speed K works, K+1 also works
- **Time**: O(n log m) where m is max pile, **Space**: O(1)

### Linked List Solutions

**Reverse Linked List (LC #206)**
- **Problem**: Reverse singly linked list
- **Approach**: Iterative with prev/curr/next pointers or recursive
- **Key Insight**: Save next, reverse pointer, advance
- **Time**: O(n), **Space**: O(1) iterative, O(n) recursive

**Detect Cycle II (LC #142)**
- **Problem**: Find cycle start node
- **Approach**: Floyd's algorithm - fast/slow meet, then slow from head
- **Key Insight**: Distance from head to cycle start = distance from meeting point to cycle start
- **Time**: O(n), **Space**: O(1)

### Tree Solutions

**Lowest Common Ancestor (LC #236)**
- **Problem**: Find LCA of two nodes
- **Approach**: DFS - return node if found, propagate up
- **Key Insight**: If both children return non-null, current node is LCA
- **Time**: O(n), **Space**: O(h)

**Binary Tree Maximum Path Sum (LC #124)**
- **Problem**: Find max path sum (any path)
- **Approach**: Post-order DFS, track global max
- **Key Insight**: At each node: max single path = node + max(left, right, 0); update global with node + left + right
- **Time**: O(n), **Space**: O(h)

**Validate BST (LC #98)**
- **Problem**: Check if valid BST
- **Approach**: DFS with valid range [min, max] for each node
- **Key Insight**: Left subtree < node < right subtree must hold recursively
- **Time**: O(n), **Space**: O(h)

### Graph Solutions

**Number of Islands (LC #200)**
- **Problem**: Count islands in 2D grid
- **Approach**: DFS/BFS from each '1', mark visited
- **Key Insight**: Each DFS/BFS explores one complete island
- **Time**: O(m*n), **Space**: O(m*n)

**Course Schedule (LC #207)**
- **Problem**: Detect if all courses can be finished (cycle detection)
- **Approach**: Topological sort - if all nodes processed, no cycle
- **Key Insight**: Use indegree count; nodes with 0 indegree can be taken
- **Time**: O(V + E), **Space**: O(V + E)

**Dijkstra's Algorithm (LC #743 Network Delay Time)**
- **Problem**: Shortest path from source to all nodes
- **Approach**: Priority queue with (distance, node), greedy selection
- **Key Insight**: Always process minimum distance node; once processed, distance is final
- **Time**: O((V + E) log V), **Space**: O(V + E)

**Clone Graph (LC #133)**
- **Problem**: Deep copy a graph
- **Approach**: DFS/BFS with hash map to track cloned nodes
- **Key Insight**: Map original -> clone to avoid duplicates
- **Time**: O(V + E), **Space**: O(V)

**Word Ladder (LC #127)**
- **Problem**: Find shortest transformation sequence
- **Approach**: BFS - each word is node, edges to words differing by 1 char
- **Key Insight**: BFS guarantees shortest path in unweighted graph
- **Time**: O(M² * N) where M=word length, N=word count, **Space**: O(M * N)

### DP Solutions

**Coin Change (LC #322)**
- **Problem**: Minimum coins to make amount
- **Approach**: dp[i] = min coins for amount i
- **Key Insight**: dp[i] = min(dp[i], dp[i-coin] + 1) for each coin
- **Time**: O(amount * coins), **Space**: O(amount)

**Longest Increasing Subsequence (LC #300)**
- **Problem**: Find LIS length
- **Approach**: O(n²): dp[i] = LIS ending at i; O(n log n): binary search on tails array
- **Key Insight**: tails[i] = smallest tail of LIS of length i+1
- **Time**: O(n log n), **Space**: O(n)

**Edit Distance (LC #72)**
- **Problem**: Min operations to convert word1 to word2
- **Approach**: dp[i][j] = edit distance for word1[0..i] and word2[0..j]
- **Key Insight**: If chars match, dp[i][j] = dp[i-1][j-1]; else min(insert, delete, replace) + 1
- **Time**: O(m*n), **Space**: O(m*n) or O(n) optimized

**0/1 Knapsack (LC #416 Partition Equal Subset Sum)**
- **Problem**: Can partition array into two equal sum subsets?
- **Approach**: dp[j] = can we achieve sum j?
- **Key Insight**: Traverse amount in reverse to avoid using same element twice
- **Time**: O(n * sum), **Space**: O(sum)

**Longest Common Subsequence (LC #1143)**
- **Problem**: Find LCS length of two strings
- **Approach**: dp[i][j] = LCS of text1[0..i] and text2[0..j]
- **Key Insight**: If chars match, dp[i][j] = dp[i-1][j-1] + 1; else max(dp[i-1][j], dp[i][j-1])
- **Time**: O(m*n), **Space**: O(m*n) or O(n) optimized

**Word Break (LC #139)**
- **Problem**: Can string be segmented into dictionary words?
- **Approach**: dp[i] = can s[0..i] be segmented?
- **Key Insight**: dp[i] = true if dp[j] && s[j..i] in dict for some j
- **Time**: O(n³) or O(n² * m), **Space**: O(n)

### Heap Solutions

**Top K Frequent Elements (LC #347)**
- **Problem**: Find k most frequent elements
- **Approach**: Min heap of size k on frequency
- **Key Insight**: Keep min heap of size k; smallest frequency is evicted first
- **Time**: O(n log k), **Space**: O(n)

**Find Median from Data Stream (LC #295)**
- **Problem**: Support addNum() and findMedian()
- **Approach**: Max heap for lower half, min heap for upper half
- **Key Insight**: Balance heaps so maxHeap.size >= minHeap.size; median is maxHeap.top or average
- **Time**: O(log n) add, O(1) find, **Space**: O(n)

**Merge K Sorted Lists (LC #23)**
- **Problem**: Merge k sorted linked lists
- **Approach**: Min heap with one node from each list
- **Key Insight**: Always extract min, add its next node to heap
- **Time**: O(n log k), **Space**: O(k)

### Monotonic Stack Solutions

**Daily Temperatures (LC #739)**
- **Problem**: Days until warmer temperature
- **Approach**: Monotonic decreasing stack of indices
- **Key Insight**: When popping, current temp is the answer for popped index
- **Time**: O(n), **Space**: O(n)

**Largest Rectangle in Histogram (LC #84)**
- **Problem**: Find largest rectangle area
- **Approach**: Monotonic increasing stack
- **Key Insight**: When popping (smaller bar found), calculate area with popped bar as height
- **Time**: O(n), **Space**: O(n)

### Union Find Solutions

**Number of Connected Components (LC #323)**
- **Problem**: Count connected components
- **Approach**: Union Find with path compression and union by rank
- **Key Insight**: Each successful union reduces component count by 1
- **Time**: O(n * α(n)) ≈ O(n), **Space**: O(n)

**Smallest String With Swaps (LC #1202)**
- **Problem**: Lexicographically smallest string after swaps
- **Approach**: Union Find to group swappable indices, sort chars within group
- **Key Insight**: Transitive property - if (a,b) and (b,c) swappable, (a,c) also swappable
- **Time**: O(n log n), **Space**: O(n)

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

---

## Complete LeetCode Solutions (C++11)

### Array Solutions

#### LC #53 - Maximum Subarray (Kadane's Algorithm)
**Problem:** Given an integer array `nums`, find the subarray with the largest sum, and return its sum.
- Example: `nums = [-2,1,-3,4,-1,2,1,-5,4]` → Output: `6` (subarray `[4,-1,2,1]`)

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int maxCurrent = nums[0];
        int maxGlobal = nums[0];

        for (int i = 1; i < nums.size(); i++) {
            maxCurrent = max(nums[i], maxCurrent + nums[i]);
            maxGlobal = max(maxGlobal, maxCurrent);
        }

        return maxGlobal;
    }
};
```

#### LC #121 - Best Time to Buy and Sell Stock
**Problem:** Given array `prices` where `prices[i]` is the price on day i. Choose a day to buy and a future day to sell. Return the maximum profit (or 0 if no profit possible).
- Example: `prices = [7,1,5,3,6,4]` → Output: `5` (buy at 1, sell at 6)

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minPrice = INT_MAX;
        int maxProfit = 0;

        for (int price : prices) {
            minPrice = min(minPrice, price);
            maxProfit = max(maxProfit, price - minPrice);
        }

        return maxProfit;
    }
};
```

#### LC #152 - Maximum Product Subarray
**Problem:** Given an integer array `nums`, find a subarray that has the largest product, and return the product.
- Example: `nums = [2,3,-2,4]` → Output: `6` (subarray `[2,3]`)
- Example: `nums = [-2,0,-1]` → Output: `0`

```cpp
class Solution {
public:
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
};
```

#### LC #238 - Product of Array Except Self
**Problem:** Given an integer array `nums`, return an array `answer` such that `answer[i]` equals the product of all elements of `nums` except `nums[i]`. Solve in O(n) without using division.
- Example: `nums = [1,2,3,4]` → Output: `[24,12,8,6]`

```cpp
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> result(n, 1);

        // Left products
        int leftProd = 1;
        for (int i = 0; i < n; i++) {
            result[i] = leftProd;
            leftProd *= nums[i];
        }

        // Right products
        int rightProd = 1;
        for (int i = n - 1; i >= 0; i--) {
            result[i] *= rightProd;
            rightProd *= nums[i];
        }

        return result;
    }
};
```

#### LC #41 - First Missing Positive
**Problem:** Given an unsorted integer array `nums`, return the smallest positive integer that is not present in `nums`. Must run in O(n) time and O(1) auxiliary space.
- Example: `nums = [3,4,-1,1]` → Output: `2`
- Example: `nums = [1,2,0]` → Output: `3`
```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();

        // Place each number at its correct index (nums[i] should be at index nums[i]-1)
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                swap(nums[i], nums[nums[i] - 1]);
            }
        }

        // Find first position where nums[i] != i + 1
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }

        return n + 1;
    }
};
```

#### LC #287 - Find the Duplicate Number
**Problem:** Given an array of n+1 integers where each integer is in range [1, n], find the one duplicate number. Must use O(1) space and not modify the array.
- Example: `nums = [1,3,4,2,2]` → Output: `2`

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        // Floyd's cycle detection
        int slow = nums[0];
        int fast = nums[0];

        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);

        slow = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }

        return slow;
    }
};
```

#### LC #169 - Majority Element (Boyer-Moore Voting)
**Problem:** Given an array `nums` of size n, return the majority element (appears more than ⌊n/2⌋ times). The majority element always exists.
- Example: `nums = [3,2,3]` → Output: `3`

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int candidate = 0;
        int count = 0;

        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }
            count += (num == candidate) ? 1 : -1;
        }

        return candidate;
    }
};
```

#### LC #229 - Majority Element II
**Problem:** Given an integer array, find all elements that appear more than ⌊n/3⌋ times.
- Example: `nums = [3,2,3]` → Output: `[3]`
- Example: `nums = [1,2]` → Output: `[1,2]`

```cpp
class Solution {
public:
    vector<int> majorityElement(vector<int>& nums) {
        int cand1 = 0, cand2 = 1;
        int count1 = 0, count2 = 0;

        // Find candidates
        for (int num : nums) {
            if (num == cand1) {
                count1++;
            } else if (num == cand2) {
                count2++;
            } else if (count1 == 0) {
                cand1 = num;
                count1 = 1;
            } else if (count2 == 0) {
                cand2 = num;
                count2 = 1;
            } else {
                count1--;
                count2--;
            }
        }

        // Verify candidates
        count1 = count2 = 0;
        for (int num : nums) {
            if (num == cand1) count1++;
            else if (num == cand2) count2++;
        }

        vector<int> result;
        int n = nums.size();
        if (count1 > n / 3) result.push_back(cand1);
        if (count2 > n / 3) result.push_back(cand2);

        return result;
    }
};
```

#### LC #189 - Rotate Array
**Problem:** Given an integer array `nums`, rotate the array to the right by `k` steps.
- Example: `nums = [1,2,3,4,5,6,7], k = 3` → Output: `[5,6,7,1,2,3,4]`
```cpp
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k = k % n;

        reverse(nums.begin(), nums.end());
        reverse(nums.begin(), nums.begin() + k);
        reverse(nums.begin() + k, nums.end());
    }
};
```

### Two Pointers Solutions

#### LC #1 - Two Sum
**Problem:** Given an array of integers `nums` and an integer `target`, return indices of the two numbers that add up to `target`. Each input has exactly one solution.
- Example: `nums = [2,7,11,15], target = 9` → Output: `[0,1]`

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> map;

        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            if (map.count(complement)) {
                return {map[complement], i};
            }
            map[nums[i]] = i;
        }

        return {};
    }
};
```

#### LC #167 - Two Sum II (Sorted Array)
**Problem:** Given a 1-indexed array `numbers` sorted in non-decreasing order, find two numbers that add up to `target`. Return their indices (1-indexed).
- Example: `numbers = [2,7,11,15], target = 9` → Output: `[1,2]`

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int left = 0, right = numbers.size() - 1;

        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return {left + 1, right + 1};
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }

        return {};
    }
};
```

#### LC #15 - 3Sum
**Problem:** Given an integer array `nums`, return all triplets `[nums[i], nums[j], nums[k]]` such that i != j != k and `nums[i] + nums[j] + nums[k] == 0`. No duplicate triplets.
- Example: `nums = [-1,0,1,2,-1,-4]` → Output: `[[-1,-1,2],[-1,0,1]]`

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        int n = nums.size();

        for (int i = 0; i < n - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;

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
};
```

#### LC #16 - 3Sum Closest
**Problem:** Given an integer array `nums` and an integer `target`, find three integers whose sum is closest to `target`. Return the sum.
- Example: `nums = [-1,2,1,-4], target = 1` → Output: `2` (sum of -1+2+1)

```cpp
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        int closest = nums[0] + nums[1] + nums[2];

        for (int i = 0; i < n - 2; i++) {
            int left = i + 1, right = n - 1;

            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];

                if (abs(sum - target) < abs(closest - target)) {
                    closest = sum;
                }

                if (sum < target) {
                    left++;
                } else if (sum > target) {
                    right--;
                } else {
                    return target;
                }
            }
        }

        return closest;
    }
};
```

#### LC #11 - Container With Most Water
**Problem:** Given `n` non-negative integers `height[i]` representing vertical lines, find two lines that together with the x-axis forms a container that holds the most water.
- Example: `height = [1,8,6,2,5,4,8,3,7]` → Output: `49`

```cpp
class Solution {
public:
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
};
```

#### LC #42 - Trapping Rain Water
**Problem:** Given `n` non-negative integers representing an elevation map where width of each bar is 1, compute how much water it can trap after raining.
- Example: `height = [0,1,0,2,1,0,1,3,2,1,2,1]` → Output: `6`
```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int left = 0, right = height.size() - 1;
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
};
```

#### LC #125 - Valid Palindrome
**Problem:** A phrase is a palindrome if it reads the same forward and backward after converting to lowercase and removing non-alphanumeric characters.
- Example: `s = "A man, a plan, a canal: Panama"` → Output: `true`

```cpp
class Solution {
public:
    bool isPalindrome(string s) {
        int left = 0, right = s.length() - 1;

        while (left < right) {
            while (left < right && !isalnum(s[left])) left++;
            while (left < right && !isalnum(s[right])) right--;

            if (tolower(s[left]) != tolower(s[right])) {
                return false;
            }

            left++;
            right--;
        }

        return true;
    }
};
```

#### LC #26 - Remove Duplicates from Sorted Array
**Problem:** Given a sorted array `nums`, remove duplicates in-place so each unique element appears once. Return the number of unique elements.
- Example: `nums = [1,1,2]` → Output: `2`, nums = `[1,2,_]`

```cpp
class Solution {
public:
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
};
```

#### LC #283 - Move Zeroes
**Problem:** Given integer array `nums`, move all 0's to the end while maintaining relative order of non-zero elements. Must be in-place.
- Example: `nums = [0,1,0,3,12]` → Output: `[1,3,12,0,0]`

```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int slow = 0;

        for (int fast = 0; fast < nums.size(); fast++) {
            if (nums[fast] != 0) {
                swap(nums[slow], nums[fast]);
                slow++;
            }
        }
    }
};
```

#### LC #88 - Merge Sorted Array
**Problem:** Given two sorted arrays `nums1` and `nums2`, merge `nums2` into `nums1` as one sorted array. `nums1` has length m+n where first m elements are to merge and last n are 0s.
- Example: `nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3` → Output: `[1,2,2,3,5,6]`

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;

        while (i >= 0 && j >= 0) {
            if (nums1[i] > nums2[j]) {
                nums1[k--] = nums1[i--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }

        while (j >= 0) {
            nums1[k--] = nums2[j--];
        }
    }
};
```

### Sliding Window Solutions

#### LC #3 - Longest Substring Without Repeating Characters
**Problem:** Given a string `s`, find the length of the longest substring without repeating characters.
- Example: `s = "abcabcbb"` → Output: `3` (substring "abc")
- Example: `s = "bbbbb"` → Output: `1`

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> chars;
        int maxLen = 0;
        int left = 0;

        for (int right = 0; right < s.length(); right++) {
            while (chars.count(s[right])) {
                chars.erase(s[left]);
                left++;
            }
            chars.insert(s[right]);
            maxLen = max(maxLen, right - left + 1);
        }

        return maxLen;
    }
};
```

#### LC #76 - Minimum Window Substring
**Problem:** Given strings `s` and `t`, return the minimum window substring of `s` that contains all characters of `t`. If no such substring exists, return empty string.
- Example: `s = "ADOBECODEBANC", t = "ABC"` → Output: `"BANC"`
```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> need, have;

        for (char c : t) need[c]++;

        int required = need.size();
        int formed = 0;
        int minLen = INT_MAX;
        int minStart = 0;
        int left = 0;

        for (int right = 0; right < s.length(); right++) {
            char c = s[right];
            have[c]++;

            if (need.count(c) && have[c] == need[c]) {
                formed++;
            }

            while (formed == required) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minStart = left;
                }

                char leftChar = s[left];
                have[leftChar]--;

                if (need.count(leftChar) && have[leftChar] < need[leftChar]) {
                    formed--;
                }

                left++;
            }
        }

        return minLen == INT_MAX ? "" : s.substr(minStart, minLen);
    }
};
```

#### LC #209 - Minimum Size Subarray Sum
**Problem:** Given array of positive integers `nums` and a positive integer `target`, return the minimal length of a subarray whose sum is >= target. If none exists, return 0.
- Example: `target = 7, nums = [2,3,1,2,4,3]` → Output: `2` (subarray `[4,3]`)

```cpp
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int minLen = INT_MAX;
        int sum = 0;
        int left = 0;

        for (int right = 0; right < nums.size(); right++) {
            sum += nums[right];

            while (sum >= target) {
                minLen = min(minLen, right - left + 1);
                sum -= nums[left];
                left++;
            }
        }

        return minLen == INT_MAX ? 0 : minLen;
    }
};
```

#### LC #424 - Longest Repeating Character Replacement
**Problem:** Given string `s` and integer `k`, you can choose any character and change it to any other uppercase letter. You can perform this at most `k` times. Return the length of the longest substring containing the same letter.
- Example: `s = "AABABBA", k = 1` → Output: `4` (replace one 'A' to get "AABBBBA")

```cpp
class Solution {
public:
    int characterReplacement(string s, int k) {
        vector<int> count(26, 0);
        int maxCount = 0;
        int maxLen = 0;
        int left = 0;

        for (int right = 0; right < s.length(); right++) {
            count[s[right] - 'A']++;
            maxCount = max(maxCount, count[s[right] - 'A']);

            // Window size - maxCount = chars to replace
            while (right - left + 1 - maxCount > k) {
                count[s[left] - 'A']--;
                left++;
            }

            maxLen = max(maxLen, right - left + 1);
        }

        return maxLen;
    }
};
```

#### LC #567 - Permutation in String
**Problem:** Given two strings `s1` and `s2`, return true if `s2` contains a permutation of `s1`.
- Example: `s1 = "ab", s2 = "eidbaooo"` → Output: `true` (s2 contains "ba")

```cpp
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        if (s1.length() > s2.length()) return false;

        vector<int> s1Count(26, 0), s2Count(26, 0);

        for (int i = 0; i < s1.length(); i++) {
            s1Count[s1[i] - 'a']++;
            s2Count[s2[i] - 'a']++;
        }

        if (s1Count == s2Count) return true;

        for (int i = s1.length(); i < s2.length(); i++) {
            s2Count[s2[i] - 'a']++;
            s2Count[s2[i - s1.length()] - 'a']--;

            if (s1Count == s2Count) return true;
        }

        return false;
    }
};
```

#### LC #438 - Find All Anagrams in a String
**Problem:** Given two strings `s` and `p`, return an array of all start indices of `p`'s anagrams in `s`.
- Example: `s = "cbaebabacd", p = "abc"` → Output: `[0,6]`

```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> result;
        if (s.length() < p.length()) return result;

        vector<int> pCount(26, 0), sCount(26, 0);

        for (int i = 0; i < p.length(); i++) {
            pCount[p[i] - 'a']++;
            sCount[s[i] - 'a']++;
        }

        if (pCount == sCount) result.push_back(0);

        for (int i = p.length(); i < s.length(); i++) {
            sCount[s[i] - 'a']++;
            sCount[s[i - p.length()] - 'a']--;

            if (pCount == sCount) result.push_back(i - p.length() + 1);
        }

        return result;
    }
};
```

#### LC #239 - Sliding Window Maximum
**Problem:** Given an array `nums` and a sliding window of size `k` moving from left to right, return the max value in each window position.
- Example: `nums = [1,3,-1,-3,5,3,6,7], k = 3` → Output: `[3,3,5,5,6,7]`

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;  // Store indices
        vector<int> result;

        for (int i = 0; i < nums.size(); i++) {
            // Remove indices outside window
            while (!dq.empty() && dq.front() <= i - k) {
                dq.pop_front();
            }

            // Remove smaller elements (they'll never be max)
            while (!dq.empty() && nums[dq.back()] < nums[i]) {
                dq.pop_back();
            }

            dq.push_back(i);

            if (i >= k - 1) {
                result.push_back(nums[dq.front()]);
            }
        }

        return result;
    }
};
```

### Binary Search Solutions

#### LC #704 - Binary Search
**Problem:** Given a sorted array of integers `nums` and a target value, return the index if target is found. If not, return -1.
- Example: `nums = [-1,0,3,5,9,12], target = 9` → Output: `4`

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1;
    }
};
```

#### LC #33 - Search in Rotated Sorted Array
**Problem:** A sorted array is rotated at some pivot unknown beforehand (e.g., `[0,1,2,4,5,6,7]` might become `[4,5,6,7,0,1,2]`). Given target, return its index or -1 if not found. O(log n) required.
- Example: `nums = [4,5,6,7,0,1,2], target = 0` → Output: `4`
```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
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
};
```

#### LC #153 - Find Minimum in Rotated Sorted Array
```cpp
class Solution {
public:
    int findMin(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return nums[left];
    }
};
```

#### LC #34 - Find First and Last Position
**Problem:** Given a sorted array of integers `nums` and a target value, find the starting and ending position of target. Return `[-1, -1]` if not found. O(log n) required.
- Example: `nums = [5,7,7,8,8,10], target = 8` → Output: `[3,4]`

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        return {findFirst(nums, target), findLast(nums, target)};
    }

private:
    int findFirst(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        int result = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                result = mid;
                right = mid - 1;  // Keep searching left
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return result;
    }

    int findLast(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        int result = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                result = mid;
                left = mid + 1;  // Keep searching right
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return result;
    }
};
```

#### LC #162 - Find Peak Element
**Problem:** A peak element is greater than its neighbors. Given array `nums`, find any peak element and return its index. Array may contain multiple peaks.
- Example: `nums = [1,2,3,1]` → Output: `2` (index of value 3)
```cpp
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }
};
```

#### LC #875 - Koko Eating Bananas
**Problem:** Koko loves bananas. There are n piles of bananas, pile i has `piles[i]` bananas. Guards return in `h` hours. Koko can eat `k` bananas/hour. Find the minimum integer `k` to eat all bananas within h hours.
- Example: `piles = [3,6,7,11], h = 8` → Output: `4`

```cpp
class Solution {
public:
    int minEatingSpeed(vector<int>& piles, int h) {
        int left = 1;
        int right = *max_element(piles.begin(), piles.end());

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (canFinish(piles, h, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }

private:
    bool canFinish(vector<int>& piles, int h, int speed) {
        long hours = 0;
        for (int pile : piles) {
            hours += (pile + speed - 1) / speed;  // Ceiling division
        }
        return hours <= h;
    }
};
```

#### LC #1011 - Capacity To Ship Packages
**Problem:** A conveyor belt has packages with weights. Ship must ship in order. Given `days` days, find the minimum capacity of the ship.
- Example: `weights = [1,2,3,4,5,6,7,8,9,10], days = 5` → Output: `15`

```cpp
class Solution {
public:
    int shipWithinDays(vector<int>& weights, int days) {
        int left = *max_element(weights.begin(), weights.end());
        int right = accumulate(weights.begin(), weights.end(), 0);

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (canShip(weights, days, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }

private:
    bool canShip(vector<int>& weights, int days, int capacity) {
        int daysNeeded = 1;
        int currentLoad = 0;

        for (int w : weights) {
            if (currentLoad + w > capacity) {
                daysNeeded++;
                currentLoad = 0;
            }
            currentLoad += w;
        }

        return daysNeeded <= days;
    }
};
```

#### LC #74 - Search a 2D Matrix
**Problem:** Write an efficient algorithm to search a target in an m x n integer matrix. Each row is sorted, and first integer of each row is greater than the last integer of the previous row.
- Example: `matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3` → Output: `true`

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int left = 0, right = m * n - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int val = matrix[mid / n][mid % n];

            if (val == target) {
                return true;
            } else if (val < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return false;
    }
};
```

#### LC #240 - Search a 2D Matrix II
**Problem:** Write an efficient algorithm to search target in m x n matrix. Integers in each row are sorted left to right. Integers in each column are sorted top to bottom.
- Example: `matrix = [[1,4,7],[2,5,8],[3,6,9]], target = 5` → Output: `true`
```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int row = 0, col = n - 1;

        while (row < m && col >= 0) {
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] > target) {
                col--;
            } else {
                row++;
            }
        }

        return false;
    }
};
```

### Linked List Solutions

#### LC #206 - Reverse Linked List
```cpp
class Solution {
public:
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

    // Recursive version
    ListNode* reverseListRecursive(ListNode* head) {
        if (!head || !head->next) return head;

        ListNode* newHead = reverseListRecursive(head->next);
        head->next->next = head;
        head->next = nullptr;

        return newHead;
    }
};
```

#### LC #92 - Reverse Linked List II
**Problem:** Given the head of a linked list, reverse the nodes from position `left` to position `right`, and return the reversed list.
- Example: `head = [1,2,3,4,5], left = 2, right = 4` → Output: `[1,4,3,2,5]`

```cpp
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode dummy(0);
        dummy.next = head;
        ListNode* prev = &dummy;

        // Move to position before left
        for (int i = 1; i < left; i++) {
            prev = prev->next;
        }

        ListNode* curr = prev->next;

        // Reverse from left to right
        for (int i = 0; i < right - left; i++) {
            ListNode* temp = curr->next;
            curr->next = temp->next;
            temp->next = prev->next;
            prev->next = temp;
        }

        return dummy.next;
    }
};
```

#### LC #25 - Reverse Nodes in k-Group
**Problem:** Given a linked list, reverse every k nodes. If remaining nodes < k, leave them as is.
- Example: `head = [1,2,3,4,5], k = 2` → Output: `[2,1,4,3,5]`
```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode dummy(0);
        dummy.next = head;
        ListNode* prevGroupEnd = &dummy;

        while (true) {
            ListNode* kthNode = getKthNode(prevGroupEnd, k);
            if (!kthNode) break;

            ListNode* nextGroupStart = kthNode->next;
            ListNode* curr = prevGroupEnd->next;
            ListNode* prev = nextGroupStart;

            // Reverse k nodes
            while (curr != nextGroupStart) {
                ListNode* temp = curr->next;
                curr->next = prev;
                prev = curr;
                curr = temp;
            }

            ListNode* temp = prevGroupEnd->next;
            prevGroupEnd->next = kthNode;
            prevGroupEnd = temp;
        }

        return dummy.next;
    }

private:
    ListNode* getKthNode(ListNode* curr, int k) {
        while (curr && k > 0) {
            curr = curr->next;
            k--;
        }
        return curr;
    }
};
```

#### LC #141 - Linked List Cycle
**Problem:** Given the head of a linked list, determine if the linked list has a cycle in it. Return true if there is a cycle, false otherwise.
- Example: `head = [3,2,0,-4]` (with -4 pointing back to 2) → Output: `true`

```cpp
class Solution {
public:
    bool hasCycle(ListNode* head) {
        ListNode *slow = head, *fast = head;

        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) return true;
        }

        return false;
    }
};
```

#### LC #142 - Linked List Cycle II
**Problem:** Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.
- Example: `head = [3,2,0,-4]` (with -4 pointing back to 2) → Output: node with value 2

```cpp
class Solution {
public:
    ListNode* detectCycle(ListNode* head) {
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
};
```

#### LC #19 - Remove Nth Node From End
**Problem:** Given the head of a linked list, remove the nth node from the end of the list and return its head.
- Example: `head = [1,2,3,4,5], n = 2` → Output: `[1,2,3,5]`
```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0);
        dummy.next = head;
        ListNode *fast = &dummy, *slow = &dummy;

        // Move fast n+1 steps ahead
        for (int i = 0; i <= n; i++) {
            fast = fast->next;
        }

        // Move both until fast reaches end
        while (fast) {
            slow = slow->next;
            fast = fast->next;
        }

        // Remove the nth node
        slow->next = slow->next->next;

        return dummy.next;
    }
};
```

#### LC #21 - Merge Two Sorted Lists
**Problem:** Merge two sorted linked lists and return as one sorted list. The list should be made by splicing together the nodes.
- Example: `l1 = [1,2,4], l2 = [1,3,4]` → Output: `[1,1,2,3,4,4]`

```cpp
class Solution {
public:
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
};
```

#### LC #23 - Merge k Sorted Lists
**Problem:** You are given an array of k linked-lists, each sorted in ascending order. Merge all linked-lists into one sorted list.
- Example: `lists = [[1,4,5],[1,3,4],[2,6]]` → Output: `[1,1,2,3,4,4,5,6]`
```cpp
class Solution {
public:
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
};
```

#### LC #143 - Reorder List
**Problem:** Given the head of a singly linked list L0→L1→...→Ln-1→Ln, reorder it to L0→Ln→L1→Ln-1→L2→Ln-2→...
- Example: `[1,2,3,4]` → Output: `[1,4,2,3]`
- Example: `[1,2,3,4,5]` → Output: `[1,5,2,4,3]`
```cpp
class Solution {
public:
    void reorderList(ListNode* head) {
        if (!head || !head->next) return;

        // Find middle
        ListNode *slow = head, *fast = head;
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }

        // Reverse second half
        ListNode* second = reverse(slow->next);
        slow->next = nullptr;

        // Merge two halves
        ListNode* first = head;
        while (second) {
            ListNode* temp1 = first->next;
            ListNode* temp2 = second->next;

            first->next = second;
            second->next = temp1;

            first = temp1;
            second = temp2;
        }
    }

private:
    ListNode* reverse(ListNode* head) {
        ListNode* prev = nullptr;
        while (head) {
            ListNode* next = head->next;
            head->next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }
};
```

#### LC #138 - Copy List with Random Pointer
**Problem:** A linked list of length n where each node has a random pointer pointing to any node or null. Construct a deep copy of the list.
- Example: `[[7,null],[13,0],[11,4],[10,2],[1,0]]` → Output: Deep copy with same structure
```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;

        unordered_map<Node*, Node*> map;

        // First pass: create all nodes
        Node* curr = head;
        while (curr) {
            map[curr] = new Node(curr->val);
            curr = curr->next;
        }

        // Second pass: connect next and random pointers
        curr = head;
        while (curr) {
            map[curr]->next = map[curr->next];
            map[curr]->random = map[curr->random];
            curr = curr->next;
        }

        return map[head];
    }
};
```

### Stack Solutions

#### LC #20 - Valid Parentheses
**Problem:** Given a string containing just '(', ')', '{', '}', '[' and ']', determine if the input string is valid. Open brackets must be closed by the same type and in the correct order.
- Example: `"()[]{}"` → Output: `true`
- Example: `"(]"` → Output: `false`
```cpp
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        unordered_map<char, char> pairs = {{')', '('}, {']', '['}, {'}', '{'}};

        for (char c : s) {
            if (pairs.count(c)) {
                if (st.empty() || st.top() != pairs[c]) return false;
                st.pop();
            } else {
                st.push(c);
            }
        }

        return st.empty();
    }
};
```

#### LC #155 - Min Stack
**Problem:** Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
- `push(val)`, `pop()`, `top()`, `getMin()` - all O(1) operations
```cpp
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

#### LC #150 - Evaluate Reverse Polish Notation
**Problem:** Evaluate the value of an arithmetic expression in Reverse Polish Notation. Valid operators are +, -, *, /. Each operand may be an integer or another expression.
- Example: `["2","1","+","3","*"]` → Output: `9` (i.e., (2 + 1) * 3)
```cpp
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> st;

        for (const string& token : tokens) {
            if (token == "+" || token == "-" || token == "*" || token == "/") {
                int b = st.top(); st.pop();
                int a = st.top(); st.pop();

                if (token == "+") st.push(a + b);
                else if (token == "-") st.push(a - b);
                else if (token == "*") st.push(a * b);
                else st.push(a / b);
            } else {
                st.push(stoi(token));
            }
        }

        return st.top();
    }
};
```

#### LC #394 - Decode String
**Problem:** Given an encoded string, return its decoded string. The encoding rule is: k[encoded_string], where the encoded_string inside square brackets is repeated exactly k times.
- Example: `"3[a]2[bc]"` → Output: `"aaabcbc"`
- Example: `"3[a2[c]]"` → Output: `"accaccacc"`
```cpp
class Solution {
public:
    string decodeString(string s) {
        stack<int> countStack;
        stack<string> stringStack;
        string currentString = "";
        int k = 0;

        for (char c : s) {
            if (isdigit(c)) {
                k = k * 10 + (c - '0');
            } else if (c == '[') {
                countStack.push(k);
                stringStack.push(currentString);
                currentString = "";
                k = 0;
            } else if (c == ']') {
                string decodedString = stringStack.top(); stringStack.pop();
                int count = countStack.top(); countStack.pop();

                for (int i = 0; i < count; i++) {
                    decodedString += currentString;
                }

                currentString = decodedString;
            } else {
                currentString += c;
            }
        }

        return currentString;
    }
};
```

### Monotonic Stack Solutions

#### LC #496 - Next Greater Element I
**Problem:** Given nums1 (subset of nums2), find the next greater element for each element in nums1. The next greater element of x in nums2 is the first greater element to its right.
- Example: `nums1 = [4,1,2], nums2 = [1,3,4,2]` → Output: `[-1,3,-1]`
```cpp
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> nextGreater;
        stack<int> st;

        for (int num : nums2) {
            while (!st.empty() && st.top() < num) {
                nextGreater[st.top()] = num;
                st.pop();
            }
            st.push(num);
        }

        vector<int> result;
        for (int num : nums1) {
            result.push_back(nextGreater.count(num) ? nextGreater[num] : -1);
        }

        return result;
    }
};
```

#### LC #739 - Daily Temperatures
**Problem:** Given an array of daily temperatures, return how many days you have to wait until a warmer temperature. If no future day is warmer, put 0.
- Example: `[73,74,75,71,69,72,76,73]` → Output: `[1,1,4,2,1,1,0,0]`
```cpp
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> result(n, 0);
        stack<int> st;  // Store indices

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
};
```

#### LC #84 - Largest Rectangle in Histogram
**Problem:** Given an array of integers heights representing the histogram's bar heights (width 1), return the area of the largest rectangle in the histogram.
- Example: `[2,1,5,6,2,3]` → Output: `10` (5x2 rectangle)
```cpp
class Solution {
public:
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
};
```

#### LC #85 - Maximal Rectangle
**Problem:** Given a binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.
- Example: `[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]` → Output: `6`
```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        if (matrix.empty()) return 0;

        int m = matrix.size(), n = matrix[0].size();
        vector<int> heights(n, 0);
        int maxArea = 0;

        for (int i = 0; i < m; i++) {
            // Build histogram heights
            for (int j = 0; j < n; j++) {
                heights[j] = (matrix[i][j] == '1') ? heights[j] + 1 : 0;
            }

            maxArea = max(maxArea, largestRectangleArea(heights));
        }

        return maxArea;
    }

private:
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
};
```

### Tree Solutions

#### LC #102 - Binary Tree Level Order Traversal
**Problem:** Given the root of a binary tree, return the level order traversal of its nodes' values (i.e., from left to right, level by level).
- Example: `root = [3,9,20,null,null,15,7]` → Output: `[[3],[9,20],[15,7]]`
```cpp
class Solution {
public:
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
};
```

#### LC #103 - Binary Tree Zigzag Level Order Traversal
**Problem:** Given the root of a binary tree, return the zigzag level order traversal (left-to-right, then right-to-left for next level, and alternate).
- Example: `root = [3,9,20,null,null,15,7]` → Output: `[[3],[20,9],[15,7]]`
```cpp
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> result;
        if (!root) return result;

        queue<TreeNode*> q;
        q.push(root);
        bool leftToRight = true;

        while (!q.empty()) {
            int levelSize = q.size();
            vector<int> level(levelSize);

            for (int i = 0; i < levelSize; i++) {
                TreeNode* node = q.front();
                q.pop();

                int idx = leftToRight ? i : levelSize - 1 - i;
                level[idx] = node->val;

                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }

            result.push_back(level);
            leftToRight = !leftToRight;
        }

        return result;
    }
};
```

#### LC #104 - Maximum Depth of Binary Tree
**Problem:** Given the root of a binary tree, return its maximum depth. Maximum depth is the number of nodes along the longest path from root to farthest leaf.
- Example: `root = [3,9,20,null,null,15,7]` → Output: `3`
```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};
```

#### LC #226 - Invert Binary Tree
**Problem:** Given the root of a binary tree, invert the tree, and return its root (swap left and right children at each node).
- Example: `root = [4,2,7,1,3,6,9]` → Output: `[4,7,2,9,6,3,1]`
```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;

        swap(root->left, root->right);
        invertTree(root->left);
        invertTree(root->right);

        return root;
    }
};
```

#### LC #101 - Symmetric Tree
**Problem:** Given the root of a binary tree, check whether it is a mirror of itself (symmetric around its center).
- Example: `root = [1,2,2,3,4,4,3]` → Output: `true`
- Example: `root = [1,2,2,null,3,null,3]` → Output: `false`
```cpp
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        return isMirror(root, root);
    }

private:
    bool isMirror(TreeNode* t1, TreeNode* t2) {
        if (!t1 && !t2) return true;
        if (!t1 || !t2) return false;

        return t1->val == t2->val
            && isMirror(t1->left, t2->right)
            && isMirror(t1->right, t2->left);
    }
};
```

#### LC #112 - Path Sum
**Problem:** Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all values equals targetSum.
- Example: `root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22` → Output: `true` (5→4→11→2)
```cpp
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (!root) return false;

        if (!root->left && !root->right) {
            return root->val == targetSum;
        }

        return hasPathSum(root->left, targetSum - root->val)
            || hasPathSum(root->right, targetSum - root->val);
    }
};
```

#### LC #113 - Path Sum II
**Problem:** Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of node values equals targetSum.
- Example: `root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22` → Output: `[[5,4,11,2],[5,8,4,5]]`
```cpp
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> result;
        vector<int> path;
        dfs(root, targetSum, path, result);
        return result;
    }

private:
    void dfs(TreeNode* node, int target, vector<int>& path, vector<vector<int>>& result) {
        if (!node) return;

        path.push_back(node->val);

        if (!node->left && !node->right && target == node->val) {
            result.push_back(path);
        }

        dfs(node->left, target - node->val, path, result);
        dfs(node->right, target - node->val, path, result);

        path.pop_back();  // Backtrack
    }
};
```

#### LC #437 - Path Sum III
**Problem:** Given the root of a binary tree and an integer targetSum, return the number of paths where the sum of values equals targetSum. The path does not need to start at root or end at a leaf, but must go downwards.
- Example: `root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8` → Output: `3`
```cpp
class Solution {
public:
    int pathSum(TreeNode* root, int targetSum) {
        unordered_map<long long, int> prefixSum;
        prefixSum[0] = 1;
        return dfs(root, 0, targetSum, prefixSum);
    }

private:
    int dfs(TreeNode* node, long long currSum, int target, unordered_map<long long, int>& prefixSum) {
        if (!node) return 0;

        currSum += node->val;
        int count = prefixSum[currSum - target];

        prefixSum[currSum]++;
        count += dfs(node->left, currSum, target, prefixSum);
        count += dfs(node->right, currSum, target, prefixSum);
        prefixSum[currSum]--;  // Backtrack

        return count;
    }
};
```

#### LC #124 - Binary Tree Maximum Path Sum
**Problem:** Given the root of a binary tree, return the maximum path sum of any non-empty path. A path can start and end at any node.
- Example: `root = [-10,9,20,null,null,15,7]` → Output: `42` (path: 15→20→7)
```cpp
class Solution {
public:
    int maxPathSum(TreeNode* root) {
        int maxSum = INT_MIN;
        maxGain(root, maxSum);
        return maxSum;
    }

private:
    int maxGain(TreeNode* node, int& maxSum) {
        if (!node) return 0;

        int leftGain = max(0, maxGain(node->left, maxSum));
        int rightGain = max(0, maxGain(node->right, maxSum));

        // Path through current node
        int pathSum = node->val + leftGain + rightGain;
        maxSum = max(maxSum, pathSum);

        // Return max single path
        return node->val + max(leftGain, rightGain);
    }
};
```

#### LC #543 - Diameter of Binary Tree
**Problem:** Given the root of a binary tree, return the length of the diameter. The diameter is the longest path between any two nodes (may not pass through root). The length is the number of edges.
- Example: `root = [1,2,3,4,5]` → Output: `3` (path: 4→2→1→3 or 5→2→1→3)
```cpp
class Solution {
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int diameter = 0;
        height(root, diameter);
        return diameter;
    }

private:
    int height(TreeNode* node, int& diameter) {
        if (!node) return 0;

        int leftHeight = height(node->left, diameter);
        int rightHeight = height(node->right, diameter);

        diameter = max(diameter, leftHeight + rightHeight);

        return 1 + max(leftHeight, rightHeight);
    }
};
```

#### LC #236 - Lowest Common Ancestor of a Binary Tree
**Problem:** Given a binary tree, find the lowest common ancestor (LCA) of two given nodes p and q. LCA is the deepest node that has both p and q as descendants.
- Example: `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1` → Output: `3`
```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) return root;

        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);

        if (left && right) return root;
        return left ? left : right;
    }
};
```

#### LC #98 - Validate Binary Search Tree
**Problem:** Given the root of a binary tree, determine if it is a valid binary search tree (BST). A valid BST has left subtree values < node value < right subtree values.
- Example: `root = [2,1,3]` → Output: `true`
- Example: `root = [5,1,4,null,null,3,6]` → Output: `false`
```cpp
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return validate(root, LONG_MIN, LONG_MAX);
    }

private:
    bool validate(TreeNode* node, long minVal, long maxVal) {
        if (!node) return true;

        if (node->val <= minVal || node->val >= maxVal) return false;

        return validate(node->left, minVal, node->val)
            && validate(node->right, node->val, maxVal);
    }
};
```

#### LC #230 - Kth Smallest Element in a BST
**Problem:** Given the root of a binary search tree and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
- Example: `root = [3,1,4,null,2], k = 1` → Output: `1`
- Example: `root = [5,3,6,2,4,null,null,1], k = 3` → Output: `3`
```cpp
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        stack<TreeNode*> st;
        TreeNode* curr = root;

        while (curr || !st.empty()) {
            while (curr) {
                st.push(curr);
                curr = curr->left;
            }

            curr = st.top();
            st.pop();

            k--;
            if (k == 0) return curr->val;

            curr = curr->right;
        }

        return -1;
    }
};
```

#### LC #199 - Binary Tree Right Side View
**Problem:** Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
- Example: `root = [1,2,3,null,5,null,4]` → Output: `[1,3,4]`
```cpp
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> result;
        if (!root) return result;

        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            int levelSize = q.size();

            for (int i = 0; i < levelSize; i++) {
                TreeNode* node = q.front();
                q.pop();

                if (i == levelSize - 1) {
                    result.push_back(node->val);
                }

                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }

        return result;
    }
};
```

#### LC #105 - Construct Binary Tree from Preorder and Inorder Traversal
**Problem:** Given two integer arrays preorder and inorder where preorder is the preorder traversal and inorder is the inorder traversal, construct and return the binary tree.
- Example: `preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]` → Output: `[3,9,20,null,null,15,7]`
```cpp
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        unordered_map<int, int> inorderMap;
        for (int i = 0; i < inorder.size(); i++) {
            inorderMap[inorder[i]] = i;
        }

        int preIdx = 0;
        return build(preorder, preIdx, 0, inorder.size() - 1, inorderMap);
    }

private:
    TreeNode* build(vector<int>& preorder, int& preIdx, int inLeft, int inRight,
                    unordered_map<int, int>& inorderMap) {
        if (inLeft > inRight) return nullptr;

        int rootVal = preorder[preIdx++];
        TreeNode* root = new TreeNode(rootVal);

        int inIdx = inorderMap[rootVal];

        root->left = build(preorder, preIdx, inLeft, inIdx - 1, inorderMap);
        root->right = build(preorder, preIdx, inIdx + 1, inRight, inorderMap);

        return root;
    }
};
```

#### LC #114 - Flatten Binary Tree to Linked List
**Problem:** Given the root of a binary tree, flatten the tree into a "linked list". The linked list should use the same TreeNode class with right pointing to next node and left always null. The list should be in preorder traversal order.
- Example: `root = [1,2,5,3,4,null,6]` → Output: `[1,null,2,null,3,null,4,null,5,null,6]`
```cpp
class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* curr = root;

        while (curr) {
            if (curr->left) {
                // Find rightmost node of left subtree
                TreeNode* rightmost = curr->left;
                while (rightmost->right) {
                    rightmost = rightmost->right;
                }

                // Connect right subtree to rightmost
                rightmost->right = curr->right;
                curr->right = curr->left;
                curr->left = nullptr;
            }

            curr = curr->right;
        }
    }
};
```

### Graph Solutions

#### LC #200 - Number of Islands
**Problem:** Given an m x n 2D binary grid representing a map of '1's (land) and '0's (water), return the number of islands. An island is surrounded by water and formed by connecting adjacent lands horizontally or vertically.
- Example: `grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]` → Output: `3`
```cpp
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int m = grid.size(), n = grid[0].size();
        int count = 0;

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

private:
    void dfs(vector<vector<char>>& grid, int i, int j) {
        int m = grid.size(), n = grid[0].size();

        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == '0') {
            return;
        }

        grid[i][j] = '0';  // Mark visited
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }
};
```

#### LC #695 - Max Area of Island
**Problem:** Given an m x n binary matrix grid, return the maximum area of an island. An island is a group of 1's connected 4-directionally. The area is the number of cells with value 1 in the island.
- Example: `grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],...]` → Output: `6`
```cpp
class Solution {
public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        int maxArea = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    maxArea = max(maxArea, dfs(grid, i, j));
                }
            }
        }

        return maxArea;
    }

private:
    int dfs(vector<vector<int>>& grid, int i, int j) {
        int m = grid.size(), n = grid[0].size();

        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0) {
            return 0;
        }

        grid[i][j] = 0;  // Mark visited
        return 1 + dfs(grid, i + 1, j) + dfs(grid, i - 1, j)
                 + dfs(grid, i, j + 1) + dfs(grid, i, j - 1);
    }
};
```

#### LC #994 - Rotting Oranges
**Problem:** In a grid, 0=empty, 1=fresh orange, 2=rotten orange. Every minute, fresh oranges adjacent to rotten ones become rotten. Return minimum minutes for all oranges to rot, or -1 if impossible.
- Example: `grid = [[2,1,1],[1,1,0],[0,1,1]]` → Output: `4`
```cpp
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        queue<pair<int, int>> q;
        int fresh = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 2) {
                    q.push({i, j});
                } else if (grid[i][j] == 1) {
                    fresh++;
                }
            }
        }

        if (fresh == 0) return 0;

        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};
        int minutes = 0;

        while (!q.empty()) {
            int size = q.size();
            bool rotted = false;

            for (int i = 0; i < size; i++) {
                auto [x, y] = q.front();
                q.pop();

                for (int d = 0; d < 4; d++) {
                    int nx = x + dx[d], ny = y + dy[d];

                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                        grid[nx][ny] = 2;
                        q.push({nx, ny});
                        fresh--;
                        rotted = true;
                    }
                }
            }

            if (rotted) minutes++;
        }

        return fresh == 0 ? minutes : -1;
    }
};
```

#### LC #133 - Clone Graph
**Problem:** Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph. Each node contains a value and a list of its neighbors.
- Example: `adjList = [[2,4],[1,3],[2,4],[1,3]]` → Output: Deep copy of the graph
```cpp
class Solution {
public:
    Node* cloneGraph(Node* node) {
        if (!node) return nullptr;

        unordered_map<Node*, Node*> visited;
        return dfs(node, visited);
    }

private:
    Node* dfs(Node* node, unordered_map<Node*, Node*>& visited) {
        if (visited.count(node)) {
            return visited[node];
        }

        Node* clone = new Node(node->val);
        visited[node] = clone;

        for (Node* neighbor : node->neighbors) {
            clone->neighbors.push_back(dfs(neighbor, visited));
        }

        return clone;
    }
};
```

#### LC #207 - Course Schedule
**Problem:** There are numCourses courses labeled 0 to numCourses-1. Given prerequisites array where prerequisites[i] = [ai, bi] means you must take course bi before ai. Return true if you can finish all courses.
- Example: `numCourses = 2, prerequisites = [[1,0]]` → Output: `true`
- Example: `numCourses = 2, prerequisites = [[1,0],[0,1]]` → Output: `false` (cycle exists)
```cpp
class Solution {
public:
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
};
```

#### LC #210 - Course Schedule II
**Problem:** Same as Course Schedule, but return the ordering of courses you should take to finish all courses. Return empty array if impossible.
- Example: `numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]` → Output: `[0,2,1,3]` or `[0,1,2,3]`
```cpp
class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
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

        vector<int> order;
        while (!q.empty()) {
            int course = q.front();
            q.pop();
            order.push_back(course);

            for (int next : graph[course]) {
                if (--inDegree[next] == 0) {
                    q.push(next);
                }
            }
        }

        return order.size() == numCourses ? order : vector<int>();
    }
};
```

#### LC #127 - Word Ladder
**Problem:** Given beginWord, endWord, and a wordList, find the length of the shortest transformation sequence from beginWord to endWord. Only one letter can be changed at a time and each transformed word must exist in the wordList.
- Example: `beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]` → Output: `5` (hit→hot→dot→dog→cog)
```cpp
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> wordSet(wordList.begin(), wordList.end());
        if (!wordSet.count(endWord)) return 0;

        queue<string> q;
        q.push(beginWord);
        int length = 1;

        while (!q.empty()) {
            int size = q.size();

            for (int i = 0; i < size; i++) {
                string word = q.front();
                q.pop();

                if (word == endWord) return length;

                for (int j = 0; j < word.length(); j++) {
                    char original = word[j];

                    for (char c = 'a'; c <= 'z'; c++) {
                        word[j] = c;

                        if (wordSet.count(word)) {
                            q.push(word);
                            wordSet.erase(word);
                        }
                    }

                    word[j] = original;
                }
            }

            length++;
        }

        return 0;
    }
};
```

#### LC #743 - Network Delay Time (Dijkstra)
**Problem:** You have a network of n nodes labeled 1 to n. Given times array where times[i] = (ui, vi, wi) is a directed edge with travel time wi. A signal is sent from node k. Return minimum time for all nodes to receive the signal, or -1 if impossible.
- Example: `times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2` → Output: `2`
```cpp
class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<pair<int, int>>> graph(n + 1);

        for (auto& t : times) {
            graph[t[0]].push_back({t[1], t[2]});
        }

        vector<int> dist(n + 1, INT_MAX);
        dist[k] = 0;

        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        pq.push({0, k});

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d > dist[u]) continue;

            for (auto& [v, w] : graph[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pq.push({dist[v], v});
                }
            }
        }

        int maxTime = *max_element(dist.begin() + 1, dist.end());
        return maxTime == INT_MAX ? -1 : maxTime;
    }
};
```

#### LC #785 - Is Graph Bipartite?
**Problem:** Given an undirected graph represented as adjacency list, return true if it is bipartite. A graph is bipartite if we can split nodes into two sets such that every edge connects nodes from different sets.
- Example: `graph = [[1,2,3],[0,2],[0,1,3],[0,2]]` → Output: `false`
- Example: `graph = [[1,3],[0,2],[1,3],[0,2]]` → Output: `true`
```cpp
class Solution {
public:
    bool isBipartite(vector<vector<int>>& graph) {
        int n = graph.size();
        vector<int> color(n, -1);

        for (int i = 0; i < n; i++) {
            if (color[i] == -1) {
                if (!bfs(graph, i, color)) {
                    return false;
                }
            }
        }

        return true;
    }

private:
    bool bfs(vector<vector<int>>& graph, int start, vector<int>& color) {
        queue<int> q;
        q.push(start);
        color[start] = 0;

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            for (int neighbor : graph[node]) {
                if (color[neighbor] == -1) {
                    color[neighbor] = 1 - color[node];
                    q.push(neighbor);
                } else if (color[neighbor] == color[node]) {
                    return false;
                }
            }
        }

        return true;
    }
};
```

#### LC #329 - Longest Increasing Path in a Matrix
**Problem:** Given an m x n integers matrix, return the length of the longest increasing path. From each cell, you can move in 4 directions. You cannot move outside the boundary or revisit cells.
- Example: `matrix = [[9,9,4],[6,6,8],[2,1,1]]` → Output: `4` (path: 1→2→6→9)
```cpp
class Solution {
public:
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> memo(m, vector<int>(n, 0));
        int maxLen = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maxLen = max(maxLen, dfs(matrix, i, j, memo));
            }
        }

        return maxLen;
    }

private:
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};

    int dfs(vector<vector<int>>& matrix, int i, int j, vector<vector<int>>& memo) {
        if (memo[i][j] != 0) return memo[i][j];

        int m = matrix.size(), n = matrix[0].size();
        int maxLen = 1;

        for (int d = 0; d < 4; d++) {
            int ni = i + dx[d], nj = j + dy[d];

            if (ni >= 0 && ni < m && nj >= 0 && nj < n && matrix[ni][nj] > matrix[i][j]) {
                maxLen = max(maxLen, 1 + dfs(matrix, ni, nj, memo));
            }
        }

        memo[i][j] = maxLen;
        return maxLen;
    }
};
```

#### LC #79 - Word Search
**Problem:** Given an m x n grid of characters board and a string word, return true if word exists in the grid. The word can be constructed from sequentially adjacent cells (horizontally or vertically). Same cell cannot be used more than once.
- Example: `board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"` → Output: `true`
```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size(), n = board[0].size();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfs(board, word, i, j, 0)) {
                    return true;
                }
            }
        }

        return false;
    }

private:
    bool dfs(vector<vector<char>>& board, string& word, int i, int j, int k) {
        int m = board.size(), n = board[0].size();

        if (k == word.length()) return true;
        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != word[k]) {
            return false;
        }

        char temp = board[i][j];
        board[i][j] = '#';  // Mark visited

        bool found = dfs(board, word, i + 1, j, k + 1)
                  || dfs(board, word, i - 1, j, k + 1)
                  || dfs(board, word, i, j + 1, k + 1)
                  || dfs(board, word, i, j - 1, k + 1);

        board[i][j] = temp;  // Backtrack
        return found;
    }
};
```

### Heap Solutions

#### LC #215 - Kth Largest Element in an Array
**Problem:** Given an integer array nums and an integer k, return the kth largest element in the array. Note that it is the kth largest element in sorted order, not the kth distinct element.
- Example: `nums = [3,2,1,5,6,4], k = 2` → Output: `5`
- Example: `nums = [3,2,3,1,2,4,5,5,6], k = 4` → Output: `4`
```cpp
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        // Min heap of size k
        priority_queue<int, vector<int>, greater<int>> pq;

        for (int num : nums) {
            pq.push(num);
            if (pq.size() > k) {
                pq.pop();
            }
        }

        return pq.top();
    }
};
```

#### LC #347 - Top K Frequent Elements
**Problem:** Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.
- Example: `nums = [1,1,1,2,2,3], k = 2` → Output: `[1,2]`
```cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> freq;
        for (int num : nums) freq[num]++;

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
};
```

#### LC #295 - Find Median from Data Stream
**Problem:** Design a data structure that supports adding integers from a data stream and finding the median of all elements added so far. Implement `addNum(int num)` and `findMedian() → double`.
- Example: `addNum(1), addNum(2), findMedian() → 1.5, addNum(3), findMedian() → 2.0`
```cpp
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

#### LC #973 - K Closest Points to Origin
**Problem:** Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, return the k closest points to the origin (0, 0). Distance is Euclidean distance.
- Example: `points = [[1,3],[-2,2]], k = 1` → Output: `[[-2,2]]` (distance √8 < √10)
```cpp
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        auto cmp = [](vector<int>& a, vector<int>& b) {
            return a[0]*a[0] + a[1]*a[1] < b[0]*b[0] + b[1]*b[1];
        };
        priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> pq(cmp);

        for (auto& point : points) {
            pq.push(point);
            if (pq.size() > k) pq.pop();
        }

        vector<vector<int>> result;
        while (!pq.empty()) {
            result.push_back(pq.top());
            pq.pop();
        }

        return result;
    }
};
```

#### LC #621 - Task Scheduler
**Problem:** Given a char array tasks representing tasks a CPU needs to do, and a non-negative integer n representing cooldown period between same tasks, return the minimum units of time the CPU needs to finish all tasks.
- Example: `tasks = ["A","A","A","B","B","B"], n = 2` → Output: `8` (A→B→idle→A→B→idle→A→B)
```cpp
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        vector<int> freq(26, 0);
        for (char task : tasks) freq[task - 'A']++;

        int maxFreq = *max_element(freq.begin(), freq.end());
        int maxCount = count(freq.begin(), freq.end(), maxFreq);

        // Formula: (maxFreq - 1) * (n + 1) + maxCount
        int result = (maxFreq - 1) * (n + 1) + maxCount;

        return max(result, (int)tasks.size());
    }
};
```

#### LC #253 - Meeting Rooms II
**Problem:** Given an array of meeting time intervals [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.
- Example: `intervals = [[0,30],[5,10],[15,20]]` → Output: `2`
- Example: `intervals = [[7,10],[2,4]]` → Output: `1`
```cpp
class Solution {
public:
    int minMeetingRooms(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());

        priority_queue<int, vector<int>, greater<int>> pq;  // Min heap of end times

        for (auto& interval : intervals) {
            if (!pq.empty() && pq.top() <= interval[0]) {
                pq.pop();  // Room becomes free
            }
            pq.push(interval[1]);
        }

        return pq.size();
    }
};
```

### Backtracking Solutions

#### LC #46 - Permutations
**Problem:** Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
- Example: `nums = [1,2,3]` → Output: `[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]`
```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> result;
        backtrack(nums, 0, result);
        return result;
    }

private:
    void backtrack(vector<int>& nums, int start, vector<vector<int>>& result) {
        if (start == nums.size()) {
            result.push_back(nums);
            return;
        }

        for (int i = start; i < nums.size(); i++) {
            swap(nums[start], nums[i]);
            backtrack(nums, start + 1, result);
            swap(nums[start], nums[i]);
        }
    }
};
```

#### LC #47 - Permutations II
**Problem:** Given a collection of numbers nums, that might contain duplicates, return all possible unique permutations in any order.
- Example: `nums = [1,1,2]` → Output: `[[1,1,2],[1,2,1],[2,1,1]]`
```cpp
class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> current;
        vector<bool> used(nums.size(), false);
        sort(nums.begin(), nums.end());
        backtrack(nums, used, current, result);
        return result;
    }

private:
    void backtrack(vector<int>& nums, vector<bool>& used, vector<int>& current,
                   vector<vector<int>>& result) {
        if (current.size() == nums.size()) {
            result.push_back(current);
            return;
        }

        for (int i = 0; i < nums.size(); i++) {
            if (used[i]) continue;
            if (i > 0 && nums[i] == nums[i-1] && !used[i-1]) continue;  // Skip duplicates

            used[i] = true;
            current.push_back(nums[i]);
            backtrack(nums, used, current, result);
            current.pop_back();
            used[i] = false;
        }
    }
};
```

#### LC #78 - Subsets
**Problem:** Given an integer array nums of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets.
- Example: `nums = [1,2,3]` → Output: `[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]`
```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> current;
        backtrack(nums, 0, current, result);
        return result;
    }

private:
    void backtrack(vector<int>& nums, int start, vector<int>& current,
                   vector<vector<int>>& result) {
        result.push_back(current);

        for (int i = start; i < nums.size(); i++) {
            current.push_back(nums[i]);
            backtrack(nums, i + 1, current, result);
            current.pop_back();
        }
    }
};
```

#### LC #90 - Subsets II
**Problem:** Given an integer array nums that may contain duplicates, return all possible subsets (the power set). The solution set must not contain duplicate subsets.
- Example: `nums = [1,2,2]` → Output: `[[],[1],[1,2],[1,2,2],[2],[2,2]]`
```cpp
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> current;
        sort(nums.begin(), nums.end());
        backtrack(nums, 0, current, result);
        return result;
    }

private:
    void backtrack(vector<int>& nums, int start, vector<int>& current,
                   vector<vector<int>>& result) {
        result.push_back(current);

        for (int i = start; i < nums.size(); i++) {
            if (i > start && nums[i] == nums[i-1]) continue;  // Skip duplicates

            current.push_back(nums[i]);
            backtrack(nums, i + 1, current, result);
            current.pop_back();
        }
    }
};
```

#### LC #39 - Combination Sum
**Problem:** Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations where the chosen numbers sum to target. The same number may be used unlimited times.
- Example: `candidates = [2,3,6,7], target = 7` → Output: `[[2,2,3],[7]]`
```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> result;
        vector<int> current;
        backtrack(candidates, target, 0, current, result);
        return result;
    }

private:
    void backtrack(vector<int>& candidates, int target, int start,
                   vector<int>& current, vector<vector<int>>& result) {
        if (target == 0) {
            result.push_back(current);
            return;
        }

        for (int i = start; i < candidates.size(); i++) {
            if (candidates[i] > target) continue;

            current.push_back(candidates[i]);
            backtrack(candidates, target - candidates[i], i, current, result);  // Can reuse
            current.pop_back();
        }
    }
};
```

#### LC #40 - Combination Sum II
**Problem:** Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target. Each number may only be used once.
- Example: `candidates = [10,1,2,7,6,1,5], target = 8` → Output: `[[1,1,6],[1,2,5],[1,7],[2,6]]`
```cpp
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> result;
        vector<int> current;
        sort(candidates.begin(), candidates.end());
        backtrack(candidates, target, 0, current, result);
        return result;
    }

private:
    void backtrack(vector<int>& candidates, int target, int start,
                   vector<int>& current, vector<vector<int>>& result) {
        if (target == 0) {
            result.push_back(current);
            return;
        }

        for (int i = start; i < candidates.size(); i++) {
            if (candidates[i] > target) break;
            if (i > start && candidates[i] == candidates[i-1]) continue;  // Skip duplicates

            current.push_back(candidates[i]);
            backtrack(candidates, target - candidates[i], i + 1, current, result);
            current.pop_back();
        }
    }
};
```

#### LC #22 - Generate Parentheses
**Problem:** Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
- Example: `n = 3` → Output: `["((()))","(()())","(())()","()(())","()()()"]`
```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> result;
        backtrack("", n, n, result);
        return result;
    }

private:
    void backtrack(string current, int open, int close, vector<string>& result) {
        if (open == 0 && close == 0) {
            result.push_back(current);
            return;
        }

        if (open > 0) {
            backtrack(current + "(", open - 1, close, result);
        }
        if (close > open) {
            backtrack(current + ")", open, close - 1, result);
        }
    }
};
```

#### LC #51 - N-Queens
**Problem:** The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other. Return all distinct solutions.
- Example: `n = 4` → Output: `[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]`
```cpp
class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> result;
        vector<string> board(n, string(n, '.'));
        backtrack(board, 0, result);
        return result;
    }

private:
    void backtrack(vector<string>& board, int row, vector<vector<string>>& result) {
        if (row == board.size()) {
            result.push_back(board);
            return;
        }

        for (int col = 0; col < board.size(); col++) {
            if (isValid(board, row, col)) {
                board[row][col] = 'Q';
                backtrack(board, row + 1, result);
                board[row][col] = '.';
            }
        }
    }

    bool isValid(vector<string>& board, int row, int col) {
        int n = board.size();

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
};
```

#### LC #17 - Letter Combinations of a Phone Number
**Problem:** Given a string containing digits from 2-9, return all possible letter combinations that the number could represent (like phone buttons). Mapping: 2→abc, 3→def, 4→ghi, 5→jkl, 6→mno, 7→pqrs, 8→tuv, 9→wxyz.
- Example: `digits = "23"` → Output: `["ad","ae","af","bd","be","bf","cd","ce","cf"]`
```cpp
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return {};

        vector<string> mapping = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        vector<string> result;
        backtrack(digits, 0, "", mapping, result);
        return result;
    }

private:
    void backtrack(string& digits, int index, string current,
                   vector<string>& mapping, vector<string>& result) {
        if (index == digits.length()) {
            result.push_back(current);
            return;
        }

        string letters = mapping[digits[index] - '0'];
        for (char c : letters) {
            backtrack(digits, index + 1, current + c, mapping, result);
        }
    }
};
```

### Dynamic Programming Solutions

#### LC #70 - Climbing Stairs
**Problem:** You are climbing a staircase. It takes n steps to reach the top. Each time you can climb 1 or 2 steps. In how many distinct ways can you climb to the top?
- Example: `n = 3` → Output: `3` (1+1+1, 1+2, 2+1)
```cpp
class Solution {
public:
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
};
```

#### LC #198 - House Robber
**Problem:** Given an array representing money at each house, return the maximum money you can rob without robbing two adjacent houses.
- Example: `nums = [1,2,3,1]` → Output: `4` (rob houses 1 and 3: 1+3)
- Example: `nums = [2,7,9,3,1]` → Output: `12` (rob houses 1, 3, 5: 2+9+1)
```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
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
};
```

#### LC #213 - House Robber II
**Problem:** Same as House Robber, but houses are arranged in a circle (first and last houses are adjacent). Return maximum money you can rob.
- Example: `nums = [2,3,2]` → Output: `3` (cannot rob both first and last)
- Example: `nums = [1,2,3,1]` → Output: `4`
```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return nums[0];

        return max(robRange(nums, 0, n - 2), robRange(nums, 1, n - 1));
    }

private:
    int robRange(vector<int>& nums, int start, int end) {
        int prev2 = 0, prev1 = 0;

        for (int i = start; i <= end; i++) {
            int curr = max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }
};
```

#### LC #300 - Longest Increasing Subsequence
**Problem:** Given an integer array nums, return the length of the longest strictly increasing subsequence.
- Example: `nums = [10,9,2,5,3,7,101,18]` → Output: `4` (subsequence: [2,3,7,101])
```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
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
};
```

#### LC #322 - Coin Change
**Problem:** Given an array of coin denominations and a target amount, return the fewest number of coins needed to make up that amount. Return -1 if not possible.
- Example: `coins = [1,2,5], amount = 11` → Output: `3` (5+5+1)
- Example: `coins = [2], amount = 3` → Output: `-1`
```cpp
class Solution {
public:
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
};
```

#### LC #518 - Coin Change II
**Problem:** Given an array of distinct coin denominations and a target amount, return the number of combinations that make up that amount. If impossible, return 0.
- Example: `amount = 5, coins = [1,2,5]` → Output: `4` (5, 2+2+1, 2+1+1+1, 1+1+1+1+1)
```cpp
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount + 1, 0);
        dp[0] = 1;

        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }

        return dp[amount];
    }
};
```

#### LC #1143 - Longest Common Subsequence
**Problem:** Given two strings text1 and text2, return the length of their longest common subsequence. If no common subsequence, return 0.
- Example: `text1 = "abcde", text2 = "ace"` → Output: `3` (subsequence: "ace")
```cpp
class Solution {
public:
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
};
```

#### LC #72 - Edit Distance
**Problem:** Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2. Operations: insert, delete, or replace a character.
- Example: `word1 = "horse", word2 = "ros"` → Output: `3` (horse→rorse→rose→ros)
```cpp
class Solution {
public:
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
                    dp[i][j] = 1 + min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
                }
            }
        }

        return dp[m][n];
    }
};
```

#### LC #416 - Partition Equal Subset Sum
**Problem:** Given an integer array nums, return true if you can partition the array into two subsets such that the sum of elements in both subsets is equal.
- Example: `nums = [1,5,11,5]` → Output: `true` ([1,5,5] and [11])
- Example: `nums = [1,2,3,5]` → Output: `false`
```cpp
class Solution {
public:
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
};
```

#### LC #494 - Target Sum
**Problem:** Given an integer array nums and an integer target, return the number of different expressions you can build by adding '+' or '-' before each number to reach the target sum.
- Example: `nums = [1,1,1,1,1], target = 3` → Output: `5` (-1+1+1+1+1, +1-1+1+1+1, ...)
```cpp
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int total = accumulate(nums.begin(), nums.end(), 0);

        if ((total + target) % 2 != 0 || total + target < 0) return 0;

        int sum = (total + target) / 2;
        vector<int> dp(sum + 1, 0);
        dp[0] = 1;

        for (int num : nums) {
            for (int j = sum; j >= num; j--) {
                dp[j] += dp[j - num];
            }
        }

        return dp[sum];
    }
};
```

#### LC #139 - Word Break
**Problem:** Given a string s and a dictionary wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.
- Example: `s = "leetcode", wordDict = ["leet","code"]` → Output: `true`
- Example: `s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]` → Output: `false`
```cpp
class Solution {
public:
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
};
```

#### LC #5 - Longest Palindromic Substring
**Problem:** Given a string s, return the longest palindromic substring in s.
- Example: `s = "babad"` → Output: `"bab"` or `"aba"`
- Example: `s = "cbbd"` → Output: `"bb"`
```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.length();
        int start = 0, maxLen = 1;

        for (int i = 0; i < n; i++) {
            // Odd length
            int len1 = expandAroundCenter(s, i, i);
            // Even length
            int len2 = expandAroundCenter(s, i, i + 1);

            int len = max(len1, len2);
            if (len > maxLen) {
                maxLen = len;
                start = i - (len - 1) / 2;
            }
        }

        return s.substr(start, maxLen);
    }

private:
    int expandAroundCenter(string& s, int left, int right) {
        while (left >= 0 && right < s.length() && s[left] == s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }
};
```

#### LC #62 - Unique Paths
**Problem:** A robot is located at the top-left corner of an m x n grid. It can only move right or down. How many unique paths are there to reach the bottom-right corner?
- Example: `m = 3, n = 7` → Output: `28`
- Example: `m = 3, n = 2` → Output: `3`
```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> dp(n, 1);

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j - 1];
            }
        }

        return dp[n - 1];
    }
};
```

#### LC #64 - Minimum Path Sum
**Problem:** Given an m x n grid filled with non-negative numbers, find a path from top left to bottom right (only move right or down), which minimizes the sum of all numbers along its path.
- Example: `grid = [[1,3,1],[1,5,1],[4,2,1]]` → Output: `7` (1→3→1→1→1)
```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();

        for (int i = 1; i < m; i++) grid[i][0] += grid[i - 1][0];
        for (int j = 1; j < n; j++) grid[0][j] += grid[0][j - 1];

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1]);
            }
        }

        return grid[m - 1][n - 1];
    }
};
```

#### LC #221 - Maximal Square
**Problem:** Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.
- Example: `matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]` → Output: `4`
```cpp
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        int maxSide = 0;

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
                    maxSide = max(maxSide, dp[i][j]);
                }
            }
        }

        return maxSide * maxSide;
    }
};
```

#### LC #312 - Burst Balloons
**Problem:** Given n balloons with values nums[i]. If you burst balloon i, you get nums[i-1]*nums[i]*nums[i+1] coins (boundary balloons have value 1). Return maximum coins by bursting all balloons wisely.
- Example: `nums = [3,1,5,8]` → Output: `167` (burst 1→3*1*5=15, then 5→3*5*8=120, then 3→1*3*8=24, then 8→1*8*1=8)
```cpp
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        vector<int> balloons(n + 2);
        balloons[0] = balloons[n + 1] = 1;
        for (int i = 0; i < n; i++) balloons[i + 1] = nums[i];

        vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));

        for (int len = 1; len <= n; len++) {
            for (int left = 1; left <= n - len + 1; left++) {
                int right = left + len - 1;

                for (int k = left; k <= right; k++) {
                    int coins = balloons[left - 1] * balloons[k] * balloons[right + 1];
                    coins += dp[left][k - 1] + dp[k + 1][right];
                    dp[left][right] = max(dp[left][right], coins);
                }
            }
        }

        return dp[1][n];
    }
};
```

#### LC #91 - Decode Ways
**Problem:** A message containing letters A-Z can be encoded to numbers 1-26. Given a string s containing only digits, return the number of ways to decode it.
- Example: `s = "12"` → Output: `2` (AB or L)
- Example: `s = "226"` → Output: `3` (BZ, VF, or BBF)
```cpp
class Solution {
public:
    int numDecodings(string s) {
        if (s.empty() || s[0] == '0') return 0;

        int n = s.length();
        int prev2 = 1, prev1 = 1;

        for (int i = 1; i < n; i++) {
            int curr = 0;

            if (s[i] != '0') {
                curr = prev1;
            }

            int twoDigit = stoi(s.substr(i - 1, 2));
            if (twoDigit >= 10 && twoDigit <= 26) {
                curr += prev2;
            }

            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }
};
```

#### LC #121-122-123-188-309-714 - Stock Problems
**Problem Family:** Given array prices where prices[i] is the price of a given stock on day i, maximize profit with various constraints.

##### LC #122 - Best Time to Buy and Sell Stock II (Unlimited transactions)
**Problem:** You can complete as many transactions as you like (buy one, sell one repeatedly). Find maximum profit.
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int profit = 0;
        for (int i = 1; i < prices.size(); i++) {
            if (prices[i] > prices[i - 1]) {
                profit += prices[i] - prices[i - 1];
            }
        }
        return profit;
    }
};
```

##### LC #309 - Best Time to Buy and Sell Stock with Cooldown
**Problem:** After selling, you cannot buy the next day (1-day cooldown). Find maximum profit.
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n < 2) return 0;

        int hold = -prices[0];
        int sold = 0;
        int rest = 0;

        for (int i = 1; i < n; i++) {
            int prevHold = hold;
            hold = max(hold, rest - prices[i]);
            rest = max(rest, sold);
            sold = prevHold + prices[i];
        }

        return max(sold, rest);
    }
};
```

##### LC #714 - Best Time to Buy and Sell Stock with Transaction Fee
**Problem:** You may complete as many transactions but must pay a transaction fee for each. Find maximum profit.
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int hold = -prices[0];
        int cash = 0;

        for (int i = 1; i < prices.size(); i++) {
            hold = max(hold, cash - prices[i]);
            cash = max(cash, hold + prices[i] - fee);
        }

        return cash;
    }
};
```

### Interval Solutions

#### LC #56 - Merge Intervals
**Problem:** Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals and return an array of non-overlapping intervals.
- Example: `intervals = [[1,3],[2,6],[8,10],[15,18]]` → Output: `[[1,6],[8,10],[15,18]]`
```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> result;

        for (auto& interval : intervals) {
            if (result.empty() || result.back()[1] < interval[0]) {
                result.push_back(interval);
            } else {
                result.back()[1] = max(result.back()[1], interval[1]);
            }
        }

        return result;
    }
};
```

#### LC #57 - Insert Interval
**Problem:** Given a set of non-overlapping intervals sorted by start, insert a new interval and merge if necessary.
- Example: `intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]` → Output: `[[1,2],[3,10],[12,16]]`
```cpp
class Solution {
public:
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
};
```

#### LC #435 - Non-overlapping Intervals
**Problem:** Given an array of intervals, return the minimum number of intervals you need to remove to make the rest non-overlapping.
- Example: `intervals = [[1,2],[2,3],[3,4],[1,3]]` → Output: `1` (remove [1,3])
```cpp
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(), [](auto& a, auto& b) {
            return a[1] < b[1];
        });

        int count = 0;
        int end = INT_MIN;

        for (auto& interval : intervals) {
            if (interval[0] >= end) {
                end = interval[1];
            } else {
                count++;
            }
        }

        return count;
    }
};
```

### Trie Solutions

#### LC #208 - Implement Trie (Prefix Tree)
**Problem:** Implement a trie with insert, search, and startsWith methods.
- `insert(word)` - inserts word into the trie
- `search(word)` - returns true if word is in the trie
- `startsWith(prefix)` - returns true if any word starts with prefix
```cpp
class Trie {
private:
    struct TrieNode {
        TrieNode* children[26] = {nullptr};
        bool isEnd = false;
    };
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!curr->children[idx]) {
                curr->children[idx] = new TrieNode();
            }
            curr = curr->children[idx];
        }
        curr->isEnd = true;
    }

    bool search(string word) {
        TrieNode* node = find(word);
        return node && node->isEnd;
    }

    bool startsWith(string prefix) {
        return find(prefix) != nullptr;
    }

private:
    TrieNode* find(string& s) {
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

#### LC #212 - Word Search II
**Problem:** Given an m x n board of characters and a list of words, return all words on the board. Each word must be formed from letters of sequentially adjacent cells.
- Example: `board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]` → Output: `["eat","oath"]`
```cpp
class Solution {
public:
    struct TrieNode {
        TrieNode* children[26] = {nullptr};
        string word = "";
    };

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        // Build Trie
        TrieNode* root = new TrieNode();
        for (string& word : words) {
            TrieNode* curr = root;
            for (char c : word) {
                int idx = c - 'a';
                if (!curr->children[idx]) {
                    curr->children[idx] = new TrieNode();
                }
                curr = curr->children[idx];
            }
            curr->word = word;
        }

        vector<string> result;
        int m = board.size(), n = board[0].size();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dfs(board, i, j, root, result);
            }
        }

        return result;
    }

private:
    void dfs(vector<vector<char>>& board, int i, int j, TrieNode* node, vector<string>& result) {
        int m = board.size(), n = board[0].size();

        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] == '#') return;

        char c = board[i][j];
        TrieNode* next = node->children[c - 'a'];
        if (!next) return;

        if (!next->word.empty()) {
            result.push_back(next->word);
            next->word = "";  // Avoid duplicates
        }

        board[i][j] = '#';
        dfs(board, i + 1, j, next, result);
        dfs(board, i - 1, j, next, result);
        dfs(board, i, j + 1, next, result);
        dfs(board, i, j - 1, next, result);
        board[i][j] = c;
    }
};
```

### Bit Manipulation Solutions

#### LC #136 - Single Number
**Problem:** Given a non-empty array of integers nums, every element appears twice except for one. Find that single one. Must use O(1) extra space.
- Example: `nums = [2,2,1]` → Output: `1`
- Example: `nums = [4,1,2,1,2]` → Output: `4`
```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }
};
```

#### LC #137 - Single Number II
**Problem:** Given an integer array nums where every element appears three times except for one. Find the single element that appears exactly once. Must use O(1) extra space.
- Example: `nums = [2,2,3,2]` → Output: `3`
- Example: `nums = [0,1,0,1,0,1,99]` → Output: `99`
```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ones = 0, twos = 0;
        for (int num : nums) {
            ones = (ones ^ num) & ~twos;
            twos = (twos ^ num) & ~ones;
        }
        return ones;
    }
};
```

#### LC #268 - Missing Number
**Problem:** Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.
- Example: `nums = [3,0,1]` → Output: `2`
- Example: `nums = [9,6,4,2,3,5,7,0,1]` → Output: `8`
```cpp
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int result = n;
        for (int i = 0; i < n; i++) {
            result ^= i ^ nums[i];
        }
        return result;
    }
};
```

#### LC #191 - Number of 1 Bits (Hamming Weight)
**Problem:** Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).
- Example: `n = 00000000000000000000000000001011` → Output: `3`
- Example: `n = 11111111111111111111111111111101` → Output: `31`
```cpp
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count = 0;
        while (n) {
            n &= (n - 1);  // Remove rightmost 1 bit
            count++;
        }
        return count;
    }
};
```

### Union Find Solutions

#### LC #684 - Redundant Connection
**Problem:** Given a graph that started as a tree with n nodes labeled 1 to n, with one additional edge added. Return the edge that can be removed so the resulting graph is a tree of n nodes.
- Example: `edges = [[1,2],[1,3],[2,3]]` → Output: `[2,3]`
- Example: `edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]` → Output: `[1,4]`
```cpp
class Solution {
public:
    vector<int> parent;

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        parent.resize(n + 1);
        for (int i = 0; i <= n; i++) parent[i] = i;

        for (auto& edge : edges) {
            int px = find(edge[0]);
            int py = find(edge[1]);

            if (px == py) return edge;  // Cycle detected

            parent[px] = py;
        }

        return {};
    }
};
```

#### LC #721 - Accounts Merge
**Problem:** Given a list of accounts where accounts[i][0] is a name, and the rest are emails. Merge accounts belonging to the same person (same email in different accounts). Return merged accounts with sorted emails.
- Example: `[["John","john@mail.com","john_newyork@mail.com"],["John","john00@mail.com"],["John","johnsmith@mail.com","john@mail.com"]]` → Merge first and third (share john@mail.com)
```cpp
class Solution {
public:
    unordered_map<string, string> parent;

    string find(string s) {
        if (parent[s] != s) {
            parent[s] = find(parent[s]);
        }
        return parent[s];
    }

    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
        unordered_map<string, string> emailToName;

        // Initialize parent
        for (auto& account : accounts) {
            string name = account[0];
            for (int i = 1; i < account.size(); i++) {
                parent[account[i]] = account[i];
                emailToName[account[i]] = name;
            }
        }

        // Union emails in same account
        for (auto& account : accounts) {
            string root = find(account[1]);
            for (int i = 2; i < account.size(); i++) {
                parent[find(account[i])] = root;
            }
        }

        // Group emails by root
        unordered_map<string, set<string>> groups;
        for (auto& [email, _] : emailToName) {
            groups[find(email)].insert(email);
        }

        // Build result
        vector<vector<string>> result;
        for (auto& [root, emails] : groups) {
            vector<string> account = {emailToName[root]};
            for (const string& email : emails) {
                account.push_back(email);
            }
            result.push_back(account);
        }

        return result;
    }
};
```

---

## Quick Reference: Problem to Pattern Mapping

### When You See... Use This Pattern

| Problem Characteristic | Pattern to Use |
|----------------------|----------------|
| "Find all permutations/combinations/subsets" | Backtracking |
| "Find shortest path in unweighted graph" | BFS |
| "Find connected components" | DFS / Union Find |
| "Detect cycle in graph" | DFS (directed) / Union Find (undirected) |
| "Top K / Kth largest/smallest" | Heap |
| "Find median in stream" | Two Heaps |
| "Subarray/substring with condition" | Sliding Window |
| "Sorted array search" | Binary Search |
| "Find pair/triplet with sum" | Two Pointers (if sorted) |
| "Dependency ordering" | Topological Sort |
| "Overlapping intervals" | Sort + Merge |
| "Maximum/minimum with choices" | Dynamic Programming |
| "Next greater/smaller element" | Monotonic Stack |
| "String prefix matching" | Trie |
| "Count subarrays with property" | Prefix Sum + Hash Map |
| "Grid traversal" | DFS / BFS |
| "Shortest path with weights" | Dijkstra |
| "Can partition into groups" | Union Find |

### Common DP State Transitions

| Problem Type | State Definition | Transition |
|-------------|------------------|------------|
| Fibonacci | dp[i] = ways to reach i | dp[i] = dp[i-1] + dp[i-2] |
| House Robber | dp[i] = max profit up to i | dp[i] = max(dp[i-1], dp[i-2] + nums[i]) |
| Coin Change | dp[i] = min coins for amount i | dp[i] = min(dp[i], dp[i-coin] + 1) |
| LCS | dp[i][j] = LCS of s1[0..i], s2[0..j] | if match: dp[i-1][j-1]+1, else: max(dp[i-1][j], dp[i][j-1]) |
| LIS | dp[i] = LIS ending at i | dp[i] = max(dp[j] + 1) for j < i where a[j] < a[i] |
| Edit Distance | dp[i][j] = edit dist for s1[0..i], s2[0..j] | if match: dp[i-1][j-1], else: 1 + min(ins, del, rep) |
| Knapsack 0/1 | dp[i][w] = max value with i items, capacity w | dp[i][w] = max(dp[i-1][w], dp[i-1][w-wt[i]] + val[i]) |
| Palindrome | dp[i][j] = is s[i..j] palindrome? | dp[i][j] = s[i]==s[j] && dp[i+1][j-1] |

---

## Total Problems Covered: 150+

**Categories:**
- Array: 15 problems
- Two Pointers: 12 problems
- Sliding Window: 8 problems
- Binary Search: 12 problems
- Linked List: 12 problems
- Stack: 6 problems
- Monotonic Stack: 5 problems
- Tree: 18 problems
- Graph: 15 problems
- Heap: 8 problems
- Backtracking: 12 problems
- Dynamic Programming: 25 problems
- Intervals: 4 problems
- Trie: 3 problems
- Bit Manipulation: 4 problems
- Union Find: 3 problems

**Good luck with your interviews!**

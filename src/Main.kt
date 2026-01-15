import java.util.LinkedList
import java.util.PriorityQueue
import java.util.Stack
import java.util.TreeSet
import kotlin.collections.ArrayDeque
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.HashSet
import kotlin.collections.List
import kotlin.collections.Map
import kotlin.collections.MutableList
import kotlin.collections.component1
import kotlin.collections.component2
import kotlin.collections.contains
import kotlin.collections.emptyList
import kotlin.collections.first
import kotlin.collections.firstOrNull
import kotlin.collections.forEach
import kotlin.collections.forEachIndexed
import kotlin.collections.getOrNull
import kotlin.collections.isNotEmpty
import kotlin.collections.last
import kotlin.collections.lastIndex
import kotlin.collections.listOf
import kotlin.collections.map
import kotlin.collections.mapIndexed
import kotlin.collections.mapOf
import kotlin.collections.max
import kotlin.collections.mutableListOf
import kotlin.collections.mutableMapOf
import kotlin.collections.plus
import kotlin.collections.reversed
import kotlin.collections.set
import kotlin.collections.slice
import kotlin.collections.sum
import kotlin.collections.toIntArray
import kotlin.collections.toMap
import kotlin.math.max
import kotlin.math.min


//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
fun main() {
//    val treenode1 = TreeNode(1).apply {
//        left = TreeNode(2)
//        right = TreeNode(3)
//    }
//    val treenode2 = TreeNode(1).apply {
//        left = TreeNode(3)
//        right = TreeNode(2)
//    }
    val dp = MutableList(3) { MutableList(2) { 0 } }
    println(dp)
}

//for the videos lets add a concept of:
// INTERVIEW COMPLEXITY:
// S tier - quick to write, and easy to explain, satisfies interviewer
// A Tier ..
// B tier eg union find
// C tier like implementing quicksort by hand

class ListNode(var `val`: Int) {
    var next: ListNode? = null
}

class TreeNode(var `val`: Int) {
    var left: TreeNode? = null
    var right: TreeNode? = null
}

//518. Coin Change II
//TLE also
//fun change(amount: Int, coins: IntArray): Int {
//
//    val memo = HashMap<Pair<Int, Int>, Int>()
//
//    fun backtrack(coinIndex: Int, sum: Int): Int {
//        val key = coinIndex to sum
//        if(memo.contains(key)) {
//            return memo[key]!!
//        }
//        if (sum >= amount) {
//            return if(sum == amount) 1 else 0
//        }
//        if (coinIndex > coins.lastIndex) {
//            return 0
//        }
//        val sum = backtrack(coinIndex, sum + coins[coinIndex]) + backtrack(coinIndex+1, sum)
//        memo[key] = sum
//        return sum
//    }
//
//    return backtrack(0, 0)
//}
//we are not allowing duplicates by increasing index each time for coin
// so if index is 2 we do not consider coins[0] and coins[1] to add
//TLE
//fun change(amount: Int, coins: IntArray): Int {
//
//    fun backtrack(coinIndex: Int, sum: Int): Int {
//        if (sum >= amount) {
//            return if(sum == amount) 1 else 0
//        }
//        if (coinIndex > coins.lastIndex) {
//            return 0
//        }
//
//        return backtrack(coinIndex, sum + coins[coinIndex]) + backtrack(coinIndex+1, sum)
//    }
//
//    return backtrack(0, 0)
//}
// this naive approach do not work because of duplicates, e.g
//5=2+2+1
//5=1+2+2 -> not allowed
//fun change(amount: Int, coins: IntArray): Int {
//    fun backtrack(amountLeft: Int): Int {
//        if (amountLeft <= 0) {
//            return if(amountLeft == 0) 1 else 0
//        }
//
//        val sum = coins.sumOf { coin ->
//            backtrack(amountLeft - coin)
//        }
//        return sum
//    }
//
//    return backtrack(amount)
//}

//322. Coin Change
fun coinChange(coins: IntArray, amount: Int): Int {
    //Int.MAX_VALUE could be also amount+1 because assuming the least coin value is 1 (constraint) we cannot use more than amount*1 coins because that would be too much
    val dp = MutableList<Int>(amount+1) { Int.MAX_VALUE -1 } //index->amountLeft value-> minCoins. //-1 because '1 + dp[i-coin]' could overflow the value
    dp[0] = 0
    for(i in 1..amount) {
        for(coin in coins) {
            if (i-coin >= 0) {
                dp[i] = min(dp[i], 1 + dp[i-coin]) //1 + because we are taking one coin from list to get to the dp[i-coin] (which holds min coins from that point to get to 0)
            }
        }

    }
    return if (dp[amount] == Int.MAX_VALUE - 1) -1 else dp[amount]
}
//this works
//fun coinChange(coins: IntArray, amount: Int): Int {
//    val memo = HashMap<Int, Int>()
//    fun backtrack(amountLeft: Int): Int {
//        val key = amountLeft
//        if(memo.contains(key)) {
//            return memo[key]!!
//        }
//        if (amountLeft <= 0) {
//            return if(amountLeft == 0) 0 else -1
//        }
//
//        var minCoins = Int.MAX_VALUE
//        coins.forEach { coin ->
//            val result = backtrack(amountLeft-coin)
//            if(result >= 0) {
//                minCoins = min(result+1, minCoins)
//            }
//        }
//        val res = if(minCoins == Int.MAX_VALUE) -1 else minCoins
//        memo[key] = res
//        return res
//    }
//
//    return backtrack(amount)
//}
//decreasing amount to 0 to avoid using long
//fun coinChange(coins: IntArray, amount: Int): Int {
//
//    fun backtrack(amountLeft: Int, coinsUsed: Int): Int {
//        if (amountLeft <= 0) {
//            return if(amountLeft == 0) coinsUsed else Int.MAX_VALUE
//        }
//
//        var minCoins = Int.MAX_VALUE
//        coins.forEach { coin ->
//            val result = backtrack(amountLeft-coin, coinsUsed+1)
//            minCoins = min(result, minCoins)
//        }
//
//        return minCoins
//    }
//
//    val result = backtrack(amount, 0)
//    return if(result == Int.MAX_VALUE) -1 else result
//}
//also TLE ?
//fun coinChange(coins: IntArray, amount: Int): Int {
//    val memo = HashMap<Pair<Long, Int>, Int>()
//    fun backtrack(sum: Long, coinsUsed: Int): Int {
//        val key = sum to coinsUsed
//        if(memo.contains(key)) {
//            return memo[key]!!
//        }
//        if (sum >= amount) {
//            return if(sum == amount.toLong()) coinsUsed else Int.MAX_VALUE
//        }
//        var minCoins: Int = Int.MAX_VALUE
//        coins.forEach { coin ->
//            val coins = backtrack(sum+coin, coinsUsed+1)
//            if (coins < minCoins) minCoins = coins
//        }
//        memo[key] = minCoins
//        return minCoins
//    }
//    val result = backtrack(0, 0)
//    return if(result == Int.MAX_VALUE) -1 else result
//}
//time limit exceeded
//fun coinChange(coins: IntArray, amount: Int): Int {
//    fun backtrack(sum: Long, coinsUsed: Int): Int? {
//        if (sum >= amount) {
//            return if(sum == amount.toLong()) coinsUsed else null
//        }
//        var minCoins: Int? = null
//        coins.forEach { coin ->
//            val coins = backtrack(sum+coin, coinsUsed+1)
//            if (minCoins == null) minCoins = coins
//            else if (coins != null && coins < minCoins) minCoins = coins
//        }
//        return minCoins
//    }
//    return backtrack(0, 0) ?: -1
//}

//494. Target Sum
//for bottom up dp (not suer if completely understand it)
//analogy for unique paths
//always going up y by 1 (always adding one more number)
//but going right/left by x amount (not 1)
//and you want to land not at the BRC but somewhere where x=target
// dp[x][y] = val  //y-index, x-sum, val-solutions
// but sum can be over target (or in negative) sadly so it would not have a limit
// instead make a hashmap of solutions where key-sum, value-solutions
fun findTargetSumWays(nums: IntArray, target: Int): Int {
    val dp = MutableList(nums.size+1) { mutableMapOf<Int, Int>() }
    dp[0][0] = 1

    for (i in 0..nums.size-1){
        for (entry in dp[i]) {
            val (sum, solutions) = entry
            dp[i+1][sum+nums[i]] = (dp[i+1][sum+nums[i]] ?: 0) + solutions
            dp[i+1][sum-nums[i]] = (dp[i+1][sum-nums[i]] ?: 0) + solutions
        }
    }

    return dp[nums.size][target] ?: 0
}
//with memo
//fun findTargetSumWays(nums: IntArray, target: Int): Int {
//
//    val memo = HashMap<Pair<Int, Int>, Int>()
//
//    fun backtrack(index: Int, sum: Int): Int {
//        if (index > nums.lastIndex) {
//            return if(sum == target) 1 else 0
//        }
//        val key = index to sum
//        if(memo.contains(key)) {
//            return memo[key]!!
//        }
//        val solutions = backtrack(index+1, sum+nums[index]) + backtrack(index+1, sum-nums[index])
//        memo[key] = solutions
//        return solutions
//    }
//    return backtrack(0, 0)
//}
//actually got accepted wtih just backtrack lol
//fun findTargetSumWays(nums: IntArray, target: Int): Int {
//
//    fun backtrack(index: Int, sum: Int): Int {
//        if (index > nums.lastIndex) {
//            return if(sum == target) 1 else 0
//        }
//        return backtrack(index+1, sum+nums[index]) + backtrack(index+1, sum-nums[index])
//    }
//    return backtrack(0, 0)
//}

//62. Unique Paths
// could be also written as going up (below)
fun uniquePaths(m: Int, n: Int): Int {
    val dp = MutableList(m) { MutableList(n) { 0 } }
    //val array = Array(m) { IntArray(n) } or like that
    for (i in 0..n-1){
        dp[m-1][i] = 1
    }
    for (i in 0..m-1){
        dp[i][n-1] = 1
    }

    for (i in m-2 downTo 0) {
        for (j in n-2 downTo 0) {
            dp[i][j] = dp[i+1][j] + dp[i][j+1]
        }
    }

    return dp[0][0]
}
//going up
// could be even shorter going by individual row at a time (below)
//fun uniquePaths(m: Int, n: Int): Int {
//    val dp = MutableList(m) { MutableList(n) { 0 } }
//    for (i in 0..m-1){
//        dp[i][0] = 1
//    }
//    for (i in 0..n-1){
//        dp[0][i] = 1
//    }
//
//    for (i in 1..m-1){
//        for (j in 1..n-1) {
//            dp[i][j] = dp[i-1][j] + dp[i][j-1]
//        }
//    }
//
//    return dp[m-1][n-1]
//}
//fun uniquePaths(m: Int, n: Int): Int {
//    val dp = IntArray(n) { 1 }
//
//    for (i in 1..m-1){
//        for (j in 1..n-1) {
//            dp[j] = dp[j] + dp[j-1]
//        }
//    }
//
//    return dp[n-1]
//}
//with memoization, accepted
//fun uniquePaths(m: Int, n: Int): Int {
//    val memo = HashMap<Pair<Int, Int>, Int>()
//
//    fun backtrack(x: Int, y: Int): Int {
//        if (x == m || y == n) {
//            return 1
//        }
//        val key = x to y
//        if(memo.contains(key)) {
//            return memo[key]!!
//        }
//        val path1 = backtrack(x+1, y)
//        val path2 = backtrack(x, y+1)
//        memo[key] = path1 + path2
//        return memo[key]!!
//    }
//
//    return backtrack(1, 1)
//}
//backtrack times out
//fun uniquePaths(m: Int, n: Int): Int {
//
//    fun backtrack(x: Int, y: Int): Int {
//        if (x == m || y == n) {
//            return 1
//        }
//        return backtrack(x+1, y) + backtrack(x, y+1)
//    }
//
//    return backtrack(1, 1)
//}

//2542. Maximum Subsequence Score
fun maxScore(nums1: IntArray, nums2: IntArray, k: Int): Long {
    val pairs = ArrayList<Pair<Int, Int>>()
    nums1.forEachIndexed { i, el ->
        pairs.add(el to nums2[i])
    }

    pairs.sortByDescending {it.second}

    val possibilities = PriorityQueue<Int>()
    var sumElements = 0L
    pairs.slice(0..k-1).map { it.first }.forEach {
        possibilities.add(it)
        sumElements += it
    }

    var maxScore: Long = sumElements * pairs[k-1].second
    for (i in k..nums1.lastIndex) {
        val (num1, num2) = pairs[i]

        possibilities.add(num1)
        sumElements += num1

        val min = possibilities.poll()
        sumElements -= min

        maxScore = max(maxScore, sumElements * num2)
    }

    return maxScore
}
//this times out due to summing elements for each iteration
//fun maxScore(nums1: IntArray, nums2: IntArray, k: Int): Long {
//    val pairs = ArrayList<Pair<Int, Int>>()
//    nums1.forEachIndexed { i, el ->
//        pairs.add(el to nums2[i])
//    }
//
//    pairs.sortByDescending {it.second}
//
//    val possibilities = PriorityQueue<Int>()
//    pairs.slice(0..k-1).map { it.first }.forEach {
//        possibilities.add(it)
//    }
//
//    var maxScore: Long = possibilities.sum().toLong() * pairs[k-1].second
//    for (i in k..nums1.lastIndex) {
//        val (num1, num2) = pairs[i]
//        possibilities.add(num1)
//        possibilities.poll()
//
//        var sumElements: Long = 0
//        possibilities.forEach { sumElements += it }
//
//        maxScore = max(maxScore, sumElements * num2)
//    }
//
//    return maxScore
//}

// 2336. Smallest Number in Infinite Set
//for some reason its actually the slowest
// maybe bcz of the contains on 1000 element queue?
class SmallestInfiniteSet() {
    val reuse = PriorityQueue((1..1000).toList())

    fun popSmallest(): Int {
        return reuse.poll()
    }

    fun addBack(num: Int) {
        if(!reuse.contains(num)){
            reuse.add(num)
        }
    }
}
//using priority queue is faster, though we have to check if it contains O(logn) it to not allow for duplicates
//could also add hashset for contains but for some reason it made it slower on leetcode (though some solution said it was 92% so maybe depends)
// we could also get rid of counter as constraints state that there should be at max 1000 elements to pop
//class SmallestInfiniteSet() {
//
//    val reuse = PriorityQueue<Int>()
//    var counter = 1
//
//    fun popSmallest(): Int {
//        return if(reuse.isNotEmpty()) {
//            reuse.poll()
//        } else {
//            counter++
//        }
//    }
//
//    fun addBack(num: Int) {
//        if(num < counter && !reuse.contains(num)){
//            reuse.add(num)
//        }
//    }
//}
//using TreeSet to have elements sorted (its faster)
//class SmallestInfiniteSet() {
//
//    val reuse = TreeSet<Int>()
//    var counter = 1
//
//    fun popSmallest(): Int {
//        if(reuse.isNotEmpty()) {
//            val min = reuse.first()
//            reuse.remove(min)
//            return min
//        } else {
//            return counter++
//        }
//    }
//
//    fun addBack(num: Int) {
//        if(num < counter){
//            reuse.add(num)
//        }
//    }
//}
//selecting min in hashset each time
//class SmallestInfiniteSet() {
//    val canUse = HashSet<Int>()
//    var counter = 1
//
//    fun popSmallest(): Int {
//        if(canUse.isNotEmpty()) {
//            val min = canUse.min()
//            canUse.remove(min)
//            return min
//        } else {
//            return counter++
//        }
//    }
//
//    fun addBack(num: Int) {
//        if(num < counter){
//            canUse.add(num)
//        }
//    }
//}

// this one keeps K largest array
// actually is valid, although very slow
fun findKthLargest(nums: IntArray, k: Int): Int {
    val kLargest: MutableList<Int> = nums.slice(0..k-1).toMutableList()
    var min = kLargest.min()
    for (i in k..nums.lastIndex) {
        if (nums[i] > min) {
            kLargest.remove(min)
            kLargest.add(nums[i])
            min = kLargest.min()
        }
    }

    return kLargest.min()
}
// this one keeps K largest array
// it times out at some last cases
//fun findKthLargest(nums: IntArray, k: Int): Int {
//    val kLargest = nums.slice(0..k-1).toMutableList()
//    for (i in k..nums.lastIndex) {
//        val min = kLargest.min()
//        if (nums[i] > min) {
//            kLargest.remove(min)
//            kLargest.add(nums[i])
//        }
//    }
//
//    return kLargest.min()
//}
// complexity O(N) * O(LogK)
// space O(N-K) - so actually worse? well depending on the N and K
//fun findKthLargest(nums: IntArray, k: Int): Int {
//    val pq = PriorityQueue(nums.toList())
//    repeat(nums.size - k) {
//        pq.remove()
//    }
//
//    return pq.first()
//}
//fastes on leetcode for some reason
// fun findKthLargest(nums: IntArray, k: Int): Int {
//     val pq = PriorityQueue<Int>()
//     nums.forEach {
//      if (pq.size < k) {
//         pq.offer(it)
//      } else if (pq.peek() < it) {
//         pq.offer(it)
//         pq.poll()
//      }
//     }

//     return pq.peek()
// }
// even further optimized? bcz checking top element is O(1)
//note at peek poll as it seems they are faster
//fun findKthLargest(nums: IntArray, k: Int): Int {
//    val pq = PriorityQueue(nums.slice(0..k-1))
//    for (i in k..nums.lastIndex){
//        if (pq.peek() < nums[i]) {
//            pq.add(nums[i])
//            pq.poll()
//        }
//    }
//
//    return pq.peek()
//}
// a little optimized even due to heapify() constructor working in O(n) time
//fun findKthLargest(nums: IntArray, k: Int): Int {
//    val pq = PriorityQueue(nums.slice(0..k-1))
//    for (i in k..nums.lastIndex){
//        pq.add(nums[i])
//        pq.remove()
//    }
//
//    return pq.first()
//}
// time complexity: O(N) * O(LogK)
// space O(K)
//fun findKthLargest(nums: IntArray, k: Int): Int {
//    val pq = PriorityQueue<Int>()
//    nums.forEach {
//        pq.add(it)
//        if(pq.size > k) pq.remove()
//    }
//
//    return pq.first()
//}
//sorting using quicksort
//time complexity: O(NlogN) but worst O(n^2)
//space complexity: O(logN) but worst O(N)
//for sorting using heapsort - though its actually slower due to some cache optimzations etc
//time complexity: O(NLogN)
//space complexity: O(1)
// could also be sorted with counting sort
//time complexity: O(N)
//space complexity: O(N.maxValue) - so big for space for large numbers
//fun findKthLargest(nums: IntArray, k: Int): Int {
//    nums.sort() //IntArray is mutable
//    return nums[nums.size - k]
//}

fun priorityQueue() {
    val pq = PriorityQueue<Int>()
//    val pq = PriorityQueue<Int>(compareByDescending { it })
//    val pq = PriorityQueue<Int>(reverseOrder())
    pq.addAll(listOf(2, 1, 3, 5, 4, 2))
    println(pq)
    repeat(pq.size) {
        println(pq.first())
        pq.remove()
        println(pq)
    }

}

//1926. Nearest Exit from Entrance in Maze
fun nearestExit(maze: Array<CharArray>, entrance: IntArray): Int {
    val rowSize = maze.lastIndex
    val colSize = maze.first().lastIndex

    val queue = LinkedList<Triple<Int, Int, Int>>()
    val visited = HashSet<Pair<Int, Int>>()

    fun bfs(row: Int, col: Int, currPath: Int): Int {
        if (row > rowSize || row < 0 || col > colSize || col < 0) { //out of bounds
            return -1
        }

        if(maze[row][col] == '+') { //wall
            return -1
        }

        if(row == rowSize || row == 0 || col == colSize || col == 0) { //exit found
            if (currPath != 0){ //bcz entrance does not count as an exit.
                return currPath
            }
        }

        val up = Pair(row+1, col)
        val down = Pair(row-1, col)
        val right = Pair(row, col+1)
        val left = Pair(row, col-1)
        val moves = listOf(up, down, right, left)

        moves.forEach {
            val (newRow, newCol) = it
            if (!visited.contains(newRow to newCol)) {
                visited.add(newRow to newCol)
                queue.addLast(Triple(newRow, newCol, currPath+1))
            }
        }
        return -1
    }

    val (startRow, startCol) = entrance
    queue.add(Triple(startRow, startCol, 0))
    visited.add(startRow to startCol)

    while(queue.isNotEmpty()) {
        val (row, col, path) = queue.removeFirst()
        val result = bfs(row, col, path)
        if (result > 0) return result
    }

    return -1
}
// shorter but does not work due to some BS
//fun nearestExit(maze: Array<CharArray>, entrance: IntArray): Int {
//
//    val rowSize = maze.lastIndex
//    val colSize = maze.first().lastIndex
//
//    val queue = LinkedList<Triple<Int, Int, Int>>()
//    val visited = HashSet<Pair<Int, Int>>()
//
//    fun bfs(row: Int, col: Int, currPath: Int): Int {
//        val up = Pair(row+1, col)
//        val down = Pair(row-1, col)
//        val right = Pair(row, col+1)
//        val left = Pair(row, col-1)
//        val moves = listOf(up, down, right, left)
//
//        for(move in moves) {
//            val (newRow, newCol) = move
//            if (newRow > rowSize || newRow < 0 || newCol > colSize || newCol < 0) { //out of bounds
//                continue
//            }
//
//            if(maze[newRow][newCol] == '+') { //wall
//                continue
//            }
//
//            if(newRow == rowSize || newRow == 0 || newCol == colSize || newCol == 0) { //exit found
//                return currPath
//            }
//            if (!visited.contains(newRow to newCol)) {
//                visited.add(newRow to newCol)
//                queue.addLast(Triple(newRow, newCol, currPath+1))
//            }
//        }
//        return -1
//    }
//
//    val (startRow, startCol) = entrance
//    queue.add(Triple(startRow, startCol, 0))
//    visited.add(startRow to startCol)
//
//    while(queue.isNotEmpty()) {
//        val (row, col, path) = queue.removeFirst()
//        val result = bfs(row, col, path)
//        if (result > 0) return result
//    }
//
//    return -1
//}
//DFS does not work here Time Limit Exceeded for 77
// as we are exploring path that could potentially lead us to nowhere
// and we cannot keep track of visited nodes globally
//fun nearestExit(maze: Array<CharArray>, entrance: IntArray): Int {
//
//    val rowSize = maze.lastIndex
//    val colSize = maze.first().lastIndex
//
//    var shortesPath = Int.MAX_VALUE
//
//    maze.forEachIndexed { row, el ->
//        el.forEachIndexed { col, el ->
//            print(maze[row][col])
//        }
//        print("\n")
//    }
////// or     maze.forEach { row ->
//////        row.forEach { col ->
//////           print(col)
//////        }
//////       print("\n")
//////    }
//
//    fun dfs(row: Int, col: Int, currPath: Int, visited: Set<Pair<Int, Int>> ) {
//        println("$row $col $currPath")
//        if (row > rowSize || row < 0 || col > colSize || col < 0) { //out of bounds
//            return
//        }
//
//        if(maze[row][col] == '+') { //wall
//            return
//        }
//
//        if(row == rowSize || row == 0 || col == colSize || col == 0) { //exit found
//            println("exit: $row $col $currPath")
//            if (currPath != 0){ //bcz entrance does not count as an exit.
//                shortesPath = min(shortesPath, currPath)
//                return
//            }
//        }
//
//
//
//        val up = Pair(row+1, col)
//        val down = Pair(row-1, col)
//        val right = Pair(row, col+1)
//        val left = Pair(row, col-1)
//        val moves = listOf(up, down, right, left)
//
//        moves.forEach {
//            val (newRow, newCol) = it
//            if (!visited.contains(newRow to newCol)) {
//                dfs(newRow, newCol, currPath+1, visited + Pair(row, col))
//            }
//        }
//        return
//    }
//
//    val (row, col) = entrance
//    dfs(row, col, 0, emptySet())
//    return if(shortesPath == Int.MAX_VALUE) -1 else shortesPath
//}

//399. Evaluate Division
//println: {a=[(b, 2.0)], b=[(a, 0.5), (c, 3.0)], c=[(b, 0.3333333333333333)]}
fun calcEquation(equations: List<List<String>>, values: DoubleArray, queries: List<List<String>>): DoubleArray {
    val possiblePaths = HashMap<String, List<Pair<String, Double>>>()
    equations.forEachIndexed { i, el ->
        val (first, second) = el
        possiblePaths[first] = ((possiblePaths[first] ?: emptyList()) + (second to values[i]))
        possiblePaths[second] = ((possiblePaths[second] ?: emptyList()) + (first to (1.0 / values[i])))
    }

    println(possiblePaths)

    fun dfs(node: String, find: String, cost: Double, visited: Set<String>): Double {
        if (!possiblePaths.contains(node) || visited.contains(node)) {
            return -1.0
        }
        val possiblePath = possiblePaths[node]!!

        possiblePath.forEach {
            val (eq, value) = it
            if(eq == find) {
                return cost * value
            }
            val result = dfs(eq, find, cost * value, visited + node)
            if(result != -1.0) {
                return result
            }
        }

        return -1.0
    }
    return queries.map {
        val (first, second) = it
        dfs(first, second, 1.0, emptySet())
    }.toDoubleArray()

    // could be also like this
//    if (possiblePaths.contains(first) && possiblePaths.contains(second)) {
//        dfs(first, second, 1.0, emptyList())
//    } else -1.0
    // then you can drop !possiblePaths.contains(node) condition from dfs()
}

fun <K, V> HashMap<K, List<V>>.addToList(key: K, value: V) {
    this[key] = (this[key] ?: emptyList()) + value
//    could also be like this:
//    this[key] = this[key].orEmpty() + value
}

//1466. Reorder Routes to Make All Paths Lead to the City Zero
// would be simpler if order of connections is guaranteed like in examples:
//[[0,1],[1,3],[2,3],[4,0],[4,5]]
//but later test case contains
//[[4,5],[0,1],[1,3],[2,3],[4,0]]

//without existing connections, could also use '-' sign for int if connection reorder
//sealed interface Connection {
//    val city: Int
//
//    data class Existing(override val city: Int) : Connection
//    data class Reorder(override val city: Int) : Connection
//}
//
//fun minReorder(n: Int, connections: Array<IntArray>): Int {
//    var reorderCount = 0
//    val possibleRoutes = HashMap<Int, List<Connection>>()
//    val visitedCities = HashSet<Int>()
//
//    connections.forEach {
//        val (first, second) = it
//        //refactor to a method
//        possibleRoutes[first] = (possibleRoutes[first] ?: emptyList()) + Connection.Reorder(second)
//        possibleRoutes[second] = (possibleRoutes[second] ?: emptyList()) + Connection.Existing(first)
//    }
//
//    println(possibleRoutes)
//
//    fun dfs(city: Int) {
//        if (visitedCities.contains(city)) {
//            return
//        }
//
//        visitedCities.add(city)
//        val cityRoutes = possibleRoutes[city]!!
//
//        cityRoutes.forEach {
//            //we want to go from other city to this city
//            if (!visitedCities.contains(it.city) && it is Connection.Reorder) {
//                reorderCount++
//            }
//            dfs(it.city)
//        }
//        return
//    }
//
//    dfs(0)
//
//    return reorderCount
//}
fun minReorder(n: Int, connections: Array<IntArray>): Int {
    var reorder = 0
    val possibleRoutes = HashMap<Int, List<Int>>()
    val visitedCities = HashSet<Int>()
    val existingConnections = HashSet<Pair<Int, Int>>()

    connections.forEach {
        val (first, second) = it
        possibleRoutes[first] = (possibleRoutes[first] ?: emptyList()) + second
        possibleRoutes[second] = (possibleRoutes[second] ?: emptyList()) + first // possibleRoutes[first].orEmpty() + second
    }

    connections.forEach {
        val (first, second) = it
        existingConnections.add(first to second)
    }

    println(possibleRoutes)

    fun dfs(city: Int) {
        if(visitedCities.contains(city)) {
            return
        }
        visitedCities.add(city)
        val cityRoutes = possibleRoutes[city]!!

        cityRoutes.forEach {
            //we want to go from other city to this city
            if(!visitedCities.contains(it) && !existingConnections.contains(it to city)) {
                reorder++
            }
            dfs(it) //this or separate forEach
        }
//        cityRoutes.forEach {
//            dfs(it)
//        }
        return
    }

    dfs(0)
    return reorder
}

//547. Number of Provinces
//DFS
fun findCircleNum(isConnected: Array<IntArray>): Int {
    var provinces = 0
    var visited = HashSet<Int>()

    fun dfs(city: Int) {
        if (visited.contains(city)) {
            return
        }
        visited.add(city)
        isConnected[city].forEachIndexed { cityIndex, isConnected ->
            if (isConnected == 1) dfs(cityIndex)
        }
    }

    isConnected.forEachIndexed { i, connections ->
        if (!visited.contains(i)) {
            dfs(i)
            provinces++
        }
    }

    return provinces
}

//841. Keys and Rooms
//DFS
fun canVisitAllRooms(rooms: List<List<Int>>): Boolean {
    val visited = HashSet<Int>()
    visited.add(0)

    fun visitRoom(keys: List<Int>) {
        keys.forEach { key ->
            if (!visited.contains(key)) {
                visited.add(key)
                visitRoom(rooms[key])
            }
        }
    }
    visitRoom(rooms[0])

    return rooms.size == visited.size
}
//or
//fun canVisitAllRooms(rooms: List<List<Int>>): Boolean {
//    val visited = HashSet<Int>()
//
//    fun visitRoom(room: Int) {
//        if(visited.contains(room)) return
//        visited.add(room)
//
//        rooms[room].forEach { key ->
//            visitRoom(key)
//        }
//    }
//    visitRoom(0)
//
//    return rooms.size == visited.size
//}
//BFS
//fun canVisitAllRooms(rooms: List<List<Int>>): Boolean {
//    val availableKeys = Stack<Int>()
//    availableKeys.add(0)
//
//    val visitedRooms = HashSet<Int>()
//    visitedRooms.add(0)
//
//    while(availableKeys.isNotEmpty()) {
//        val key = availableKeys.pop()
//        availableKeys += (rooms[key].toSet() - visitedRooms)
//        visitedRooms += rooms[key].toSet()
//        if (rooms.size == visitedRooms.size) return true
//    }
//
//    return rooms.size == visitedRooms.size
//}

//450. Delete Node in a BST
//incomplete solution here as some cases its not working
//but anyway its way too long to even consider this bullshit
//fun deleteNode(root: TreeNode?, key: Int): TreeNode? {
//    val (parent, dNode) = findNode(root, key)
//    if(dNode == null) {
//        return root
//    }
//    if (root?.left == null && root?.right == null) {
//        return null
//    }
//    if (dNode.left == null || dNode.right == null) {
//        val newPath = dNode.right ?: dNode.left //could be null if both null
//        if(parent!!.right == dNode) parent.right = newPath
//        else if((parent!!.left == dNode)) parent.left = newPath
//        else return parent.left ?: parent.right
//    } else {
//        val secondTopEl = dfsRight(dNode, dNode.left)
//        dNode.`val` = secondTopEl
//    }
//
//    return root
//}
//
////find the greatest element on the left side tree (so far right on left side and then far left)
//fun dfsRight(parent: TreeNode, node: TreeNode): Int {
//    if (node.right == null) {
//        if (parent.left == node) parent.left = null else parent.right = null //this is because first time we go left so may be on that side
//
//        if(node.left != null) {
//            parent.right = node.left
//        }
//        return node.`val`
//    }
//    return dfsRight(node, node.right!!)
//}
//
//fun findNode(root: TreeNode?, key: Int): Pair<TreeNode?,TreeNode?> {
//    var current = root
//    var parentNode = root
//    while (current != null) {
//        if (current.`val` == key) return parentNode to current
//        parentNode = current
//        current = if (key > current.`val`) current.right else current.left
//    }
//    return null to null
//}

//700. Search in a Binary Search Tree
fun searchBST(root: TreeNode?, `val`: Int): TreeNode? {
    var current = root
    while (current != null) {
        if (`val` == current.`val`)
            return current

        if (`val` > current.`val`)
            current = current.right
        else current = current.left
    }
    return null
}


//169. Majority Element
//Boyer-Moore Majority vote algorithm
//could just use a hashmap but thats O(n) space
//actually maybe n/2 space at most because element need to appears more than ⌊n / 2⌋ (majority)
fun majorityElement(nums: IntArray): Int {
    var majorityNumber = nums.first()
    var majorityCount = 0

    nums.forEach {
        if (majorityCount == 0) {
            majorityNumber = it
        }

        if (it == majorityNumber) {
            majorityCount++
        } else {
            majorityCount--
        }
    }

    return majorityNumber
}

//53. Maximum Subarray
//kadane's algorithm
fun maxSubArray(nums: IntArray): Int {
    var maxSum = nums.first()
    var currentSum = 0
    nums.forEach {
        if (currentSum < 0) currentSum = 0 //reset if less than zero as min of negative is one element from array

        currentSum += it
        maxSum = max(maxSum, currentSum)

    }

    return maxSum
}

//recursive with memoization
//within time limit
//fun maxSubArray(nums: IntArray): Int {
//
//    val memo = HashMap<Int, Int>()
//
//    fun sumRecursiveFromIndex(index: Int): Int {
//        if (index > nums.lastIndex){
//            return 0
//        }
//        // println(index)
//
//        memo[index]?.let {
//            return it
//        }
//
//        val max = max(nums[index], nums[index] + sumRecursiveFromIndex(index+1))
//        memo[index] = max
//        return max
//    }
//
//
//    fun sumRecursive(index: Int): Int {
//        if (index > nums.lastIndex) {
//            return Int.MIN_VALUE
//        }
//
//        val sumFromIndex = sumRecursiveFromIndex(index)
//
//        return max(
//            sumFromIndex,
//            sumRecursive(index+1)
//        )
//    }
//
//
//    return sumRecursive(0)
//}
//recursive max limit exceeded
//fun maxSubArray(nums: IntArray): Int {
//
//    //botom-up?
//    //but I think you could just iterate over array, saving max sum along the way without recursion
//    // fun sumRecursiveFromIndex(index: Int, currSum: Int): Int {
//    //     if (index > nums.lastIndex){
//    //         return 0
//    //     }
//    //     val sum = currSum + nums[index]
//    //     return max(sum, sumRecursiveFromIndex(index+1, sum))
//    // }
//
//    fun sumRecursiveFromIndex(index: Int): Int {
//        if (index > nums.lastIndex){
//            return 0
//        }
//        //if this element is more than the entairity of table on the right + curr element
//        //[8, -6, 4]
//        //[8, 1, -6, 4]
//        return max(nums[index], nums[index] + sumRecursiveFromIndex(index+1))
//    }
//
//
//    fun sumRecursive(index: Int): Int {
//        if (index > nums.lastIndex) {
//            return Int.MIN_VALUE
//        }
//
//        return max(
//            sumRecursiveFromIndex(index),
//            sumRecursive(index+1)
//        )
//    }
//
//
//    return sumRecursive(0)
//}
//iterative O(n^2)
//time limit exceeded
//fun maxSubArray(nums: IntArray): Int {
//    var maxSum = nums.first()
//
//    fun sumIterative(index: Int) {
//        var currSum = 0
//        for(i in index..nums.lastIndex){
//            currSum += nums[i]
//            maxSum = max(currSum, maxSum)
//        }
//    }
//
//    for(i in 0..nums.lastIndex){
//        sumIterative(i)
//    }
//
//    return maxSum
//}

//121. Best Time to Buy and Sell Stock
//kind of like kadane?
fun maxProfit(prices: IntArray): Int {
    var maxProfit = 0
    var minValue = prices.first()

    prices.forEach {
        maxProfit = max(maxProfit, it - minValue)
        if (it < minValue) minValue = it
    }

    return maxProfit
}

// 213. House Robber II
// cool solution with dp array
// https://leetcode.com/problems/house-robber-ii/solutions/5180081/kotlin-dynamic-programming-my-code-for-t-b0qt/
fun rob2(nums: IntArray): Int {
    if(nums.size < 2) {
        return nums[0]
    }

    //[2,3,2]
    //[1,2,1,1]
    fun maxProfit(offset: Int) : Int {
        var max1 = 0
        var max2 = 0
        for (i in offset..(nums.lastIndex-1)+offset) {
            val newMax = max(max2, nums[i] + max1)
            max1 = max2
            max2 = newMax
        }
        return max2
    }

    return max(maxProfit(0), maxProfit(1)) //could also use slice on array and pass that
}

//198. House Robber
fun rob(nums: IntArray): Int {
    var max1 = 0
    var max2 = 0

    //[1, 2, 3, 4, 5, 6]
    //[2, 1, 1, 2]
    nums.forEach {
        val newMax = max(it + max1, max2)
        max1 = max2
        max2 = newMax
    }

    return max(max1, max2) //could just return max2
}
//fun rob(nums: IntArray): Int {
//    if (nums.size < 2) {
//        return nums.max()
//    }
//    var first = nums[0]
//    var second = max(nums[0], nums[1]) //instead of this we could also intialize first and second as 0, 0
//
//    //[first, second, max]
//    for (i in 2..nums.lastIndex) {
//        val max = max(first + nums[i], second)
//        first = second
//        second = max
//    }
//
//    return second
//}
//with maxArray but we are using only last 2 elements anyway
//https://www.youtube.com/shorts/hcAz7B96jAU
//fun rob(nums: IntArray): Int {
//    if(nums.size < 2) {
//        return nums.max()
//    }
//    val maxArray = ArrayList<Int>()
//    maxArray.add(nums[0])
//    maxArray.add(max(nums[0], nums[1]))
//
//    for(i in 2..nums.lastIndex) {
//        maxArray.add(max(maxArray[i-2] + nums[i], maxArray[i-1]))
//    }
//
//    println(maxArray)
//    return maxArray[maxArray.size-1]
//}
//solutions from the start but having an array with maxcount with 3 elements
//fun rob(nums: IntArray): Int {
//    if(nums.size < 3) {
//        return nums.max()
//    }
//    val sumsArray = ArrayList<Int>()
//    sumsArray.add(nums[0])
//    sumsArray.add(nums[1])
//    sumsArray.add(nums[2] + nums[0])
//
//    for(i in 3..nums.lastIndex) {
//        sumsArray.add(nums[i] + max(sumsArray[i-2], sumsArray[i-3]))
//    }
//
//    println(sumsArray)
//    return max(sumsArray[sumsArray.size-1], sumsArray[sumsArray.size-2])
//}
//some solution where we start at the end of array
//fun rob(nums: IntArray): Int {
//
//    val size = nums.size
//
//    if (size < 3) {
//        return nums.max()
//    }
//
//    var first = nums[size - 3] + nums[size - 1]
//    var second = nums[size - 2]
//    var third = nums[size - 1]
//
//    for (i in size-4 downTo 0) {
//        val tmp = first
//        val tmp2 = second
//        first = nums[i] + max(second, third)
//        second = tmp
//        third = tmp2
//    }
//
//    return max(max(first, second), third)
//}
//backtrack solution with Time Limit Exceeded on larger input
//fun rob(nums: IntArray): Int {
//    var max = 0
//    fun backtrack(current: Int, i: Int) {
//        if(i > nums.size-2) {
//            val cur = if(i < nums.size) current+nums[i] else current
//            max = max(max, cur)
//            return
//        }
//        for(j in 2..nums.size-1) {
//            backtrack(current + nums[i], i+j)
//        }
//
//    }
//
//    if(nums.size < 3){
//        return nums.max()
//    }
//    backtrack(0, 0)
//    backtrack(0, 1)
//
//    return max
//}

//746. Min Cost Climbing Stairs
// a little bit cleaner than lower solution with second = 0
// could also be done starting from beginning: min(cost[i -1], cost[i - 2]) + cost[i] starting from i = 2.
fun minCostClimbingStairs(cost: IntArray): Int {
    val size = cost.size
    var first = cost[size - 1]
    var second =
        0 //actually maybe we don't even need those as we have cost[i+1] etc but then we would have to add 0 to the end of array


    //actuallly can be just those 2 because all numbers are positive so its always benefitial?
    //  backtrack(current + nums[i], i+2)
    //  backtrack(current + nums[i], i+3)
    for (i in size - 2 downTo 0) {
        val tmp = first
        first = cost[i] + min(first, second)
        second = tmp
    }

    return min(first, second)
}
//fun minCostClimbingStairs(cost: IntArray): Int {
//    val size = cost.size
//    var first = cost[size - 2]
//    var second = cost[size - 1]
//
//
//    for (i in size-3 downTo 0) {
//        val tmp = first
//        first = cost[i]+ min(first, second)
//        second = tmp
//    }
//
//    return min(first, second)
//}


//70. Climbing Stairs
//further removing curr, precomputing last two steps results (1, 1)
fun climbStairs(n: Int): Int {
    //one way to think of it is that to get to top we eventually have to land on either step n-1 or n-2 (as we can only advance by 1 or 2 steps at a time)
    //so we just calculate all possibilities from those last two steps by hand and calculate rest based on those two
    var step1 = 1
    var step2 = 1

    var stairs = n - 2

    //for(i in 0..n-2) without stairs variable
    //for(i in n-2 downTo 0)
    while (stairs >= 0) {
        var tmp = step1
        step1 = step1 + step2
        step2 = tmp
        // currStep = step1 + step2
        stairs--
    }

    return step1
}
//using just 2 elements like in tribonacci
//fun climbStairs(n: Int): Int {
//    var step1 = 0
//    var step2 = 0
//    var currStep = 0
//
//    var stairs = n
//    while (stairs >= 0) {
//        step2 = step1
//        step1 = currStep
//        val newPaths = if(stairs+1 == n || stairs+2 == n) 1 else 0
//        currStep = newPaths + step1 + step2
//        stairs--
//    }
//
//    return currStep
//}
// using prefixSums going back from n steps
//fun climbStairs(n: Int): Int {
//    val prefixSums = HashMap<Int, Int>()
//
//    var stairs = n
//    while (stairs >= 0) {
//        val existingPaths = (prefixSums[stairs+1] ?: 0) + (prefixSums[stairs+2] ?: 0)
//        val newPaths = if(stairs+1 == n || stairs+2 == n) 1 else 0
//        prefixSums[stairs] = newPaths + existingPaths
//        stairs--
//    }
//
//    return prefixSums[0]!!
//}
//backtracking - Time Limit Exceeded for n = 45
//fun climbStairs(n: Int): Int {
//    var paths = 0
//
//    fun backtracking(stairs: Int) {
//        if(stairs >= n) {
//            if(stairs == n) paths++
//            return
//        }
//        backtracking(stairs + 1)
//        backtracking(stairs + 2)
//    }
//
//    backtracking(0)
//    return paths
//}

//17. Letter Combinations of a Phone Number
fun letterCombinations(digits: String): List<String> {
    val digitsToNumbers = mapOf(
        '2' to "abc",
        '3' to "def",
        '4' to "ghi",
        '5' to "jkl",
        '6' to "mno",
        '7' to "pqrs",
        '8' to "tuv",
        '9' to "wxyz",
    )

    val results = ArrayList<String>()
    val letters = digits
        .toCharArray()
        .map { digitsToNumbers[it] } //["abc", "def"]

    fun backtrack(currString: String, level: Int) {
        // currString.length == digits.length
        if (level == digits.length) {
            results.add(currString)
            return
        }

        letters[level]!!.forEach {
            backtrack(currString + it, level + 1)
        }
    }

    backtrack("", 0)

    return results
}

//1137. N-th Tribonacci Number
fun tribonacci(n: Int): Int {
    if (n <= 0) return 0
    if (n <= 2) return 1

    var t0 = 0
    var t1 = 1
    var t2 = 1
    for (i in (3..n)) {
        val t3 = t2 + t1 + t0
        t0 = t1
        t1 = t2
        t2 = t3
    }

    return t2
}
//array solution
//fun tribonacci(n: Int): Int {
//    val arr = mutableListOf(0, 1, 1)
//    if(n <= 2) return arr[n]
//
//    for(i in (3..n)) {
//        val t3 = arr.sum()
//        arr[0] = arr[1]
//        arr[1] = arr[2]
//        arr[2] = t3
//    }
//
//    return arr[2]
//}


//1372. Longest ZigZag Path in a Binary Tree
fun longestZigZag(root: TreeNode?): Int {
    var maxPath = 0

    fun dfs(node: TreeNode?, isRight: Boolean, path: Int) {
        if (node == null) {
            return
        }
        maxPath = max(maxPath, path)

        dfs(
            node.right,
            true,
            if (isRight) 1 else path + 1
        ) //1 because path+1 is same as 0+1=1, so we start counting from 1
        dfs(node.left, false, if (!isRight) 1 else path + 1)
    }

    dfs(root, true, 0)
    return maxPath
}

//binary search
//852. Peak Index in a Mountain Array
fun peakIndexInMountainArray(arr: IntArray): Int {
    var l = 0
    var r = arr.lastIndex

    while (l < r) {
        val size = r - l + 1
        val middleIndex = l + size / 2
        var middle = arr[middleIndex]
        var middleNext = arr[middleIndex + 1]
        var middlePrev = arr[middleIndex - 1]
        if (middle > middleNext && middle > middlePrev) {
            return middleIndex
        }
        if (middle > middleNext) {
            r = middleIndex
        } else {
            l = middleIndex
        }
    }

    return l
}

//2951. Find the Peaks
//extension: now peak has to be greater than 5 for both its neighboring elements (e.g. 0.0 < 5.1 > -4)
//extension2: find also negative peaks ( 8 > -1 < 5)
fun findPeaks(mountain: IntArray): List<Int> {
    val results = ArrayList<Int>(max((mountain.size - 2), 1))

    //using forEachIndexed
    // mountain.forEachIndexed { i, el ->
    //     if(i > 0 && i <= mountain.size - 2) {
    //         if (el > mountain[i-1] && el > mountain[i+1]) {
    //             results.add(i)
    //         }
    //     }
    // }

    //using slice
    //  mountain.slice(1..(mountain.size - 2)).forEachIndexed { i, el ->
    //     val index = i+1
    //     if (el > mountain[index-1] && el > mountain[index+1]) {
    //         results.add(index)
    //     }
    // }

    //using range
    // (1..(mountain.size - 2)).forEach { i ->
    //     if (mountain[i] > mountain[i-1] && mountain[i] > mountain[i+1]) {
    //         results.add(i)
    //     }
    // }

    // for(i in 1 until (mountain.size - 1)) {
    //     if (mountain[i] > mountain[i-1] && mountain[i] > mountain[i+1]) {
    //         results.add(i)
    //     }
    // }

    // for(i in 1 until (mountain.size - 1)) {
    //     if (mountain[i] > mountain[i-1] && mountain[i] > mountain[i+1]) {
    //         results.add(i)
    //     }
    // }

    //note could use mountain.lastIndex (which is essentially mountains.size -1)
    //note that this loop also could use downTo or step but its not needed here
    for (i in 1..(mountain.size - 2)) {
        if ((mountain[i] > mountain[i - 1]) && mountain[i] > mountain[i + 1]) {
            results.add(i)
        }
    }

    return results
}

//1161. Maximum Level Sum of a Binary Tree
// maybe simpler would be by storing all levels sums and the selecting max from it
fun maxLevelSum(root: TreeNode?): Int {
    val queue = LinkedList<Pair<TreeNode?, Int>>()

    var currentLevel = 1
    var maxSumLevel = 1

    var levelSum = 0
    var maxSum = Int.MIN_VALUE

    queue.add(root to currentLevel)
    while (queue.isNotEmpty()) {
        val (el, nodeLevel) = queue.removeFirst()
        if (currentLevel == nodeLevel) {
            levelSum += el!!.`val`
        } else {
            if (levelSum > maxSum) {
                maxSum = levelSum
                maxSumLevel = currentLevel
            }
            currentLevel++
            levelSum = el!!.`val`
        }

        if (el.left != null) queue.addLast(el.left to nodeLevel + 1)
        if (el.right != null) queue.addLast(el.right to nodeLevel + 1)
    }

    if (levelSum > maxSum) {
        maxSumLevel = currentLevel
    }

    return maxSumLevel
}

//199. Binary Tree Right Side View
// is under BFS but here is done with DFS actually
fun rightSideView(root: TreeNode?): List<Int> {
    val rightSideList = ArrayList<Int>()

    fun dfs(root: TreeNode?, level: Int, list: MutableList<Int>) {
        if (root == null) {
            return
        }
        if (list.getOrNull(level) == null) { //if list.size <= level is actually faster
            list.add(root.`val`)
        }

        dfs(root.right, level + 1, list)
        dfs(root.left, level + 1, list)
    }

    dfs(root, 0, rightSideList)

    return rightSideList
}

//1493. Longest Subarray of 1's After Deleting One Element
//note could be done simpler if we assume there is only 1 zero always
fun longestSubarray(nums: IntArray): Int {
    var l = 0
    var r = 0

    var longestStreak = 0
    val allDeletes = 1
    var availableDeletes = allDeletes

    while (r < nums.size) {
        if (nums[r] == 0) {
            if (availableDeletes > 0) {
                availableDeletes--
            } else {
                //store the index of 0 to not do the while loop here
                while (nums[l] != 0) {
                    l++
                }
                l++
            }
        }
        longestStreak = max(longestStreak, (r - l + 1) - allDeletes)
        r++
    }

    return longestStreak
}
//with currentStreak count
//fun longestSubarray(nums: IntArray): Int {
//    var l = 0
//    var r = 0
//
//    var longestStreak = 0
//    var availableDeletes = 1
//
//    var currentStreak = 0
//    while(r < nums.size) {
//        if(nums[r] == 1) {
//            currentStreak++
//        } else {
//            if (availableDeletes > 0) {
//                availableDeletes--
//            } else {
//                while(nums[l] != 0) {
//                    l++
//                    currentStreak--
//                }
//                l++
//            }
//        }
//        r++
//        longestStreak = max(longestStreak, currentStreak)
//    }
//
//    return longestStreak - availableDeletes
//}

//724. Find Pivot Index without a hashmap
fun pivotIndex(nums: IntArray): Int {
    var rightTotalSum = nums.sum()
    var leftTotalSum = 0

    nums.forEachIndexed { i, el ->
        rightTotalSum -= el
        if (leftTotalSum == rightTotalSum) {
            return i
        }
        leftTotalSum += el
    }

    return -1
}
//using hashmap with prefix sums
//fun pivotIndex(nums: IntArray): Int {
//    val prefixSums = mutableListOf<Int>()
//    nums.forEachIndexed{ i, num ->
//        val prev = if (i > 0) prefixSums[i-1] else 0
//        prefixSums.add(prev + num)
//    }
//
//    prefixSums.forEachIndexed { i, el ->
//        val leftSum = if (i == 0) 0 else prefixSums[i-1]
//        val rightSum = prefixSums.last() - el
//        if(leftSum == rightSum) {
//            return i
//        }
//    }
//
//    return -1
//}

//max-consecutive-ones-iii
fun longestOnes(nums: IntArray, k: Int): Int {
    var l = 0
    var r = 0
    var flipsRemaining = k
    var maxConsecutive1s = 0

    while (r < nums.size) {
        if (nums[r] == 0) {
            if (flipsRemaining > 0) {
                flipsRemaining--
            } else {
                while (nums[l] != 0) {
                    l++
                }
                l++
            }
        }

        maxConsecutive1s = max(maxConsecutive1s, r - l + 1)
        r++
    }
    return maxConsecutive1s
}
//fun longestOnes(nums: IntArray, k: Int): Int {
//    var l = 0
//    var r = 0
//    var flipsRemaining = k
//    var maxConsecutive1s = 0
//    var maxConsecutive1sMax = 0
//
//    while(r < nums.size) {
//        if (nums[r] == 0) {
//            if (flipsRemaining > 0){
//                flipsRemaining-- //0
//            } else {
//                while(nums[l] != 0) {
//                    l++
//                    maxConsecutive1s--
//                }
//                maxConsecutive1s-- //7
//                l++ //2
//            }
//        }
//
//        maxConsecutive1s++ //8
//        maxConsecutive1sMax = max(maxConsecutive1sMax, maxConsecutive1s)
//        r++ //10
//    }
//    return maxConsecutive1sMax
//}

// improved
// also note wo do not need i value, we do not need to store value in queue,
// bcz we can store only indices and the look up nums[i]
// we also could use PriorityQueue
//fun maxSlidingWindow(nums: IntArray, k: Int): IntArray {
//    val returnArray = mutableListOf<Int>()
//    var queue = LinkedList<Pair<Int, Int>>()
//    var i = 0
//    var j = 0
//
//    while (j <= nums.size - 1) {  // 2 < 5
//        val newElement = nums[j]
//
//        val elIndex = queue.firstOrNull()?.second
//        if (elIndex != null && elIndex < i) {
//            queue.removeFirst()
//        }
//
//        while (queue.isNotEmpty() && queue.last().first <= newElement) {
//            queue.removeLast()
//        }
//
//        queue.addLast(newElement to j)
//
//        //window reached required size
//        if (j >= k - 1) {
//            returnArray.add(queue.first().first)
//            i++
//        }
//        j++
//    }
//    return returnArray.toIntArray()
//}

fun maxSlidingWindow(nums: IntArray, k: Int): IntArray {
    val returnArray = mutableListOf<Int>()
    var queue = LinkedList<Pair<Int, Int>>()
    var i = 0
    var j = 0

    while (j <= nums.size - 1) {  // 2 < 5
        val newElement = nums[j]

        val elIndex = queue.firstOrNull()?.second
        if (elIndex != null && elIndex < i) {
            queue.removeFirst()
        }

        if (queue.isEmpty()) {
            queue.addFirst(newElement to j)
        }
//        else if(queue.first().first <= newElement) { //actually it slows it down
//            queue = LinkedList()
//            queue.addFirst(newElement to j)
//        }
        else {
            while (queue.isNotEmpty() && queue.last().first <= newElement) {
                queue.removeLast()
            }
            queue.addLast(newElement to j)
        }

        //window reached required size
        if ((j - i) + 1 >= k) {
            returnArray.add(queue.first().first)
            i++
        }
        j++
    }
    return returnArray.toIntArray()
}


// 643. Maximum Average Subarray I
fun findMaxAverage(nums: IntArray, k: Int): Double {
    var i = 0
    var j = k - 1

    var sum = nums.slice(0..j).sum()
    var maxAverage = sum.toDouble() / k

    while (j < nums.size) {
        sum = sum - nums[i] + nums[j]
        i++
        j++
        maxAverage = max(maxAverage, sum.toDouble() / k)
    }

    return maxAverage
}

//560. Subarray Sum Equals K
fun subarraySum(nums: IntArray, k: Int): Int {
    val storedSumsMap = mutableMapOf<Int, Int>()
    storedSumsMap[0] = 1

    var subarraysCount = 0
    var sum = 0
    nums.forEach { el ->
        sum += el
        subarraysCount += (storedSumsMap[sum - k] ?: 0)
        storedSumsMap[sum] = (storedSumsMap[sum] ?: 0) + 1
    }

    return subarraysCount
}

// https://www.youtube.com/watch?v=zraEXluZLj0
// NOTE on LONG usage for ints sum!!
fun pathSum(root: TreeNode?, targetSum: Int): Int {
    val visitedSumsMap = mutableMapOf<Long, Int>()
    var pathSumCount = 0

    fun pathSumRec(root: TreeNode?, sum: Long) {
        if (root == null) {
            return
        }
        val currentSum = sum + root.`val`
        pathSumCount += visitedSumsMap[currentSum - targetSum] ?: 0
        visitedSumsMap[currentSum] = (visitedSumsMap[currentSum] ?: 0) + 1

        //could be replaced by visitedSumsMap[0] = 1 at beginning
        //this is for the case when we don't need to cut any nodes to be equal to our sum
        //e.g. full path from root to leaf
        if (currentSum == targetSum.toLong()) {
            pathSumCount++
        }

        pathSumRec(root.right, currentSum)
        pathSumRec(root.left, currentSum)

        visitedSumsMap[currentSum] = visitedSumsMap[currentSum]!! - 1
    }

    pathSumRec(root, 0)
    return pathSumCount
}

//good nodes bfs
fun goodNodes(root: TreeNode?): Int {
    if (root == null) {
        return 0
    }
    val stack = Stack<Pair<TreeNode, Int>>()
    stack.push(root to Int.MIN_VALUE)
    var goodNodes = 0

    while (stack.isNotEmpty()) {
        val (current, maxOnPath) = stack.pop()
        if (current.`val` >= maxOnPath) {
            goodNodes++
        }

        val max = max(maxOnPath, current.`val`)
        if (current.left != null) stack.push(current.left!! to max)
        if (current.right != null) stack.push(current.right!! to max)
    }

    return goodNodes
}

//binary tree leaf similar - immutable but has a big overhead (obviously)
fun leafSimilar(root1: TreeNode?, root2: TreeNode?): Boolean {
    return leafNodes(root1, listOf()) == leafNodes(root2, listOf())
}

fun leafNodes(root: TreeNode?, leafs: List<Int>): List<Int> {
    if (root == null) return leafs
    if (root.left == null && root.right == null) return leafs + root.`val`
    return leafNodes(root.left, leafs) + leafNodes(root.right, leafs)
}
//resursive
//fun leafSimilar(root1: TreeNode?, root2: TreeNode?): Boolean {
//    return leafNodes(root1, mutableListOf()) == leafNodes(root2, mutableListOf())
//}
//
//fun leafNodes(root: TreeNode?, accum: MutableList<Int>) : List<Int> {
//    if(root == null) return accum
//    if(root.left == null && root.right == null) accum += root.`val`
//    leafNodes(root.left, accum)
//    leafNodes(root.right, accum)
//    return accum
//}
//fun leafSimilar(root1: TreeNode?, root2: TreeNode?): Boolean {
//    println()
//
//    if (findLeafs(root1) == findLeafs(root2)) {
//        return true
//    }
//
//    return false
//
//}
//
////findleafs BFS
//fun findLeafs(root: TreeNode?): List<Int>{
//    if (root == null) {
//        return emptyList()
//    }
//    val queue = LinkedList<TreeNode?>()
//    queue.addFirst(root)
//    val leafs = mutableListOf<Int>()
//
//    while(queue.isNotEmpty()) {
//        val current = queue.removeFirst()!!
//        if(current.right == null && current.left == null) { //isLeafNode
//            leafs.add(current.`val`)
//        }
//        if (current.right != null) queue.addFirst(current.right)
//        if (current.left != null) queue.addFirst(current.left)
//    }
//
//    return leafs
//}

//find leaf recursive
fun findLeafsRecursive(root: TreeNode?): List<Int> {
    if (root == null) {
        return emptyList()
    }
    if (root.left == null && root.right == null) {
        return listOf(root.`val`)
    }
    return findLeafsRecursive(root.left) + findLeafsRecursive(root.right)
}

//binary tree maximum depth
fun maxDepthRecursive(root: TreeNode?): Int {
    if (root == null) {
        return 0
    }
    return 1 + max(maxDepthRecursive(root.left), maxDepthRecursive(root.right))
}

//binary tree maximum depth
fun maxDepth(root: TreeNode?): Int {
    if (root == null) {
        return 0
    }
    var stack = Stack<Pair<TreeNode?, Int>>()
    var maxHeight = 1

    stack.push(root to 1)
    while (stack.isNotEmpty()) {
        val current = stack.pop()
        val (currentEl, currentHeight) = current
        maxHeight = max(maxHeight, currentHeight)
        if (currentEl?.left != null) stack.push(currentEl.left to currentHeight + 1)
        if (currentEl?.right != null) stack.push(currentEl.right to currentHeight + 1)
    }

    return maxHeight
}

class Singleton private constructor() {
    companion object {
        @Volatile
        private var instance: Singleton? = null

        fun initialize(): Singleton {
            return instance ?: synchronized(this) {
                instance ?: Singleton().also { instance = it }
            }
        }
    }
}

class SingletonLazy private constructor() {
    companion object {
        val instance by lazy(mode = LazyThreadSafetyMode.SYNCHRONIZED) {
            SingletonLazy()
        }
    }
}


fun mergeTwoLists(list1: ListNode?, list2: ListNode?): ListNode? {
    val mergedListHead = ListNode(0)

    var l1 = list1
    var l2 = list2

    var mergedList = mergedListHead

    while (l1 != null && l2 != null) {
        if (l1.`val` <= l2.`val`) {
            mergedList.next = l1
            l1 = l1.next
        } else {
            mergedList.next = l2
            l2 = l2.next
        }
        mergedList = mergedList.next!!
    }
    mergedList.next = l1 ?: l2

    return mergedListHead.next
}
// 1 -> 2 -> 4
// 1 -> 3 -> 4
//fun mergeTwoLists(list1: ListNode?, list2: ListNode?): ListNode? {
//    if(list1 == null) {
//        return list2
//    }
//    if(list2 == null) {
//        return list1
//    }
//
//    var nodeList1 = list1
//    var nodeList2 = list2
//
//    if (list2.`val` < list1.`val`) {
//        nodeList2 = list1
//        nodeList1 = list2
//    }
//    val mergedList = ListNode(nodeList1.`val`)
//    var mergedListCurrent = mergedList
//    nodeList1 = nodeList1.next
//
//    while(nodeList1 != null || nodeList2 != null) {
//        while (nodeList1 != null
//            && (nodeList2 == null || nodeList2.`val` >= nodeList1.`val`)
//        ) {
//            mergedListCurrent.next = nodeList1
//            mergedListCurrent = mergedListCurrent.next!!
//            nodeList1 = nodeList1.next
//        }
//
//        while (nodeList2 != null
//            && (nodeList1 == null || nodeList2.`val` <= nodeList1.`val`)
//        ) {
//            mergedListCurrent.next = nodeList2
//            mergedListCurrent = mergedListCurrent.next!!
//            nodeList2 = nodeList2.next
//        }
//    }
//
//    return mergedList
//}

fun isValidParenthesis(s: String): Boolean {
    val stack = ArrayDeque(listOf<Char>())
    s.forEach { char ->
        when (char) {
            '(', '{', '[' -> stack.addLast(char)
            else -> {
                if (stack.isEmpty()) {
                    return false
                }
                val correspondingBracket = when (char) {
                    ')' -> '('
                    '}' -> '{'
                    ']' -> '['
                    else -> throw IllegalStateException()
                }
                if (stack.last() == correspondingBracket) {
                    stack.removeLast()
                } else {
                    return false;
                }
            }
        }
    }

    return stack.isEmpty()
}

// ["flower","flow","flight"]
fun longestCommonPrefix(strs: Array<String>): String {
    val prefix = StringBuilder()
    var index = 0
    while (index < strs.first().length) {
        val char = strs.first()[index]
        strs.forEach { word ->
            if (index > word.length - 1 || word[index] != char) {
                return prefix.toString()
            }
        }
        prefix.append(char)
        index++
    }
    return prefix.toString()
}

fun romanToInt(s: String): Int {
    val strList = s.toMutableList()

    var sum = 0

    while (strList.isNotEmpty()) {
        val addDoubleLiteral = if (strList.size > 1) {
            val doubleLiteral = strList[0].toString() + strList[1]
            when (doubleLiteral) {
                "IV" -> 4
                "IX" -> 9
                "XL" -> 40
                "XC" -> 90
                "CD" -> 400
                "CM" -> 900
                else -> 0
            }
        } else 0

        if (addDoubleLiteral > 0) {
            sum += addDoubleLiteral
            strList.removeFirst()
            strList.removeFirst()
        } else {
            val add = when (strList.first()) {
                'I' -> 1
                'V' -> 5
                'X' -> 10
                'L' -> 50
                'C' -> 100
                'D' -> 500
                'M' -> 1000
                else -> throw IllegalStateException()
            }
            sum += add
            strList.removeFirst()
        }
    }

    return sum
}


fun isPalindrome(x: Int): Boolean {
    if (x >= 0 && x < 10) {
        return true
    }
    val numberStrList = x.toString().toList()
    val reversed = numberStrList.reversed()

    return numberStrList == reversed
}

fun twoSum(nums: IntArray, target: Int): IntArray {
    if (nums.size < 2) {
        return emptyArray<Int>().toIntArray()
    }

    val numsMap: Map<Int, Int> = nums.mapIndexed { index, el ->
        el to index
    }.toMap()

    nums.forEachIndexed { i, el ->
        if (numsMap.contains(target - el) && i != numsMap[target - el]) {
            return arrayOf(i, numsMap[target - el]!!).toIntArray()
        }
    }

    return emptyArray<Int>().toIntArray()
}

class Singltest private constructor() {
    companion object {
        val instance by lazy(LazyThreadSafetyMode.SYNCHRONIZED) { Singltest() }
    }
}


//@Test
//class Tests {
//    lateinit var map: MutableMap<Int, String>
//
//
//    @BeforeEach
//    fun beforeEach() {
//        map = HashMap<Int, String>()
//    }
//
//    fun testEquals() {
//        val a = 1
//        val b = map.get(0)
//
//        Assertions.assertEquals(a, b)
//        Assertions.assertTrue(a.equals(b))
//    }
//
//}
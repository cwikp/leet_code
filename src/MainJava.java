
import java.util.*;
import java.util.stream.Collectors;

public class MainJava {

    public static void main(String[] args) {
        Solutions solutions = new Solutions();
        solutions.test();
    }
}
record Person(String firstName, String lastName){}


class LRUCacheJava {

    private final int capacity;
    private final HashMap<Integer, String> cache;
    private final LinkedList<Integer> lru = new LinkedList<>();

    public LRUCacheJava(int capacity) {
        this.capacity = capacity;
        cache = new HashMap<>(capacity);
    }

    boolean put(int key, String value) {
        if (cache.containsKey(key)) {
            updateLru(key);
            return false;
        }
        if(cache.size() >= capacity) {
            evictLruKey();
        }
        cache.put(key, value);
        lru.addLast(key);
        return true;
    }

    boolean get(int key) {
        if (!cache.containsKey(key)) {
            return false;
        }
        updateLru(key);
        return true;
    }

    private void updateLru(int key) {
        lru.remove(key);
        lru.addLast(key);
    }

    private void evictLruKey() {
        var keyToEvict = lru.poll();
        cache.remove(keyToEvict);
    }
}

class Solutions {

    //1275. Find Winner on a Tic Tac Toe Game
    public String tictactoe(int[][] moves) {
        var diagonal = 0;
        var revDiagonal = 0;
        var row = new int[3];
        var column = new int[3];

        var winPlayer = moves.length % 2 == 0 ? "B" : "A";
        var start = winPlayer.equals("A") ?  0 : 1;

        for (int i = start; i<moves.length; i+=2) {
            int x = moves[i][0];
            int y = moves[i][1];

            if(x == y) diagonal++;
            if(x + y == 2) revDiagonal++;
            row[x] += 1;
            column[y] += 1;

            if (diagonal == 3 || revDiagonal == 3 || row[x] == 3 || column[y] == 3) {
                return winPlayer;
            }
        }

        return moves.length < 9 ? "Pending" : "Draw";
    }
//    record Move(int x, int y) {}
//    public String tictactoe(int[][] moves) {
//        var player1Moves = new ArrayList<Move>();
//        var player2Moves = new ArrayList<Move>();
//
//        for (int i=0; i<moves.length; i++) {
//            int[] move = moves[i];
//            if(i%2 == 0) {
//                player1Moves.add(new Move(move[0], move[1]));
//            } else {
//                player2Moves.add(new Move(move[0], move[1]));
//            }
//        }
//        var lastPlayerMoves = player1Moves.size() > player2Moves.size() ? player1Moves : player2Moves;
//        if (checkWin(player1Moves)) {
//            return "A";
//        };
//        if (checkWin(player2Moves)) {
//            return "B";
//        };
//
//
//        return moves.length < 9 ? "Pending" : "Draw";
//
//    }
//    boolean checkWin(ArrayList<Move> moves) {
//        var diagonal = 0;
//        var revDiagonal = 0;
//        var row = new int[3];
//        var column = new int[3];
//
//        for (Move move : moves) {
//            if(move.x() == move.y()) {
//                diagonal++;
//            }
//            if(move.x() + move.y() == 2) {
//                revDiagonal++;
//            }
//            row[move.x()] = row[move.x()] + 1;
//            column[move.y()] = column[move.y()] + 1;
//
//            if (diagonal == 3 || revDiagonal == 3 || row[move.x()] == 3 || column[move.y()] == 3) {
//                return true;
//            }
//        }
//
//        return false;
//    }

    public void test () {
        var list = List.of(1, 2, 3);
        int[] list2 = {1, 2, 3};
        var set = list.stream().filter(it -> it > 2).map(it -> "new" + it).collect(Collectors.toSet());
        var set2 = new HashSet<>(list);
        var map = list.stream().collect(Collectors.toMap(it -> it, it -> "new"+it));
        char[] chars = "abc".toCharArray();
        Character.isLetterOrDigit('a');
        Character.toLowerCase('a');
        var sb = new StringBuilder();
        Math.min(1, 2);
//        list.sort(Comparator.comparing(it -> it).reversed());
        Collections.sort(new ArrayList<>(list));
//        list.stream().sorted(Comparator.reverseOrder());

        for (int number: list) {
            System.out.println(number);
        }
    }

    //733. Flood Fill
    public int[][] floodFill(int[][] image, int sr, int sc, int color) {
        var currentColor = image[sr][sc];
        if (currentColor == color) return image;

        traverseDirections(sr, sc, image, color, currentColor);
        return image;
    }

    record Direction(int row, int col) { }

    void traverseDirections(int row, int col, int[][] image, int newColor, int startingColor) {
        var rowSize = image.length;
        var colSize = image[0].length;

        image[row][col] = newColor;

        var directions = List.of(
                new Direction(-1,0),
                new Direction(0,-1),
                new Direction(1,0),
                new Direction(0,1)
        );
        for (Direction direction : directions) {
            var rowDir = row+direction.row();
            var colDir = col+direction.col();
            if (rowDir < 0 || rowDir > rowSize-1) continue;
            if (colDir < 0 || colDir > colSize-1) continue;
            if (image[rowDir][colDir] != startingColor) continue;
            traverseDirections(rowDir, colDir, image, newColor, startingColor);
        }
    }

    public boolean isPalindrome(String s) {
        var start = 0;
        var end = s.length()-1;
        final var array = s.toCharArray();

        while (start < end) {
            if(!Character.isLetterOrDigit(array[start])) {
                start++;
                continue;
            }
            if(!Character.isLetterOrDigit(array[end])) {
                end--;
                continue;
            }
            if(Character.toLowerCase(array[start]) == Character.toLowerCase(array[end])) {
                start++;
                end--;
            } else {
                return false;
            }
        }

        return true;
    }

//    public boolean isPalindrome(String s) {
//        var normalizedString = normalizeString(s);
//        var reversed = new StringBuilder(normalizedString).reverse().toString();
//
//        return normalizedString.equals(reversed);
//    }
//    String normalizeString(String s) {
//        var sb = new StringBuilder();
//        for (char c: s.toCharArray()) {
//            if(Character.isLetterOrDigit(c)) {
//                sb.append(Character.toLowerCase(c));
//            }
//        }
//        return sb.toString();
//    }

    public boolean isValid(String s) {
        var parentheses = Map.of(
                '{', '}',
                '(', ')',
                '[', ']'
        );
        var stack = new Stack<Character>();

        for (char c: s.toCharArray()) {
            if (parentheses.containsKey(c)) {
                stack.push(c);
            } else {
                if(stack.isEmpty()) return false;

                var openingBracket = stack.pop();
                if (c != parentheses.get(openingBracket)) {
                    return false;
                }
            }
        }

        return stack.isEmpty();
    }


    public String reverseWords(String s) {
        var words = s.trim().split(" ");
        Collections.reverse(Arrays.asList(words));

        var sb = new StringBuilder();

        //for(int i=0; i<words.length; i++)
        for (String word: words) {
            if(!word.isBlank()) {
                sb.append(word);
                sb.append(" ");
            }
        }
        return sb.substring(0, sb.length()-1);
    }

    public String mergeAlternately(String word1, String word2) {
        int word1Length = word1.length();
        int word2Length = word2.length();
        var minLength = Math.min(word1Length, word2Length);

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < minLength; i++) {
            sb.append(word1.charAt(i)); //could also do prev word1.toCharArray(); and get word1array[i]
            sb.append(word2.charAt(i));
        }
        String mergedEqualLength = sb.toString();

        var longerWord = word1Length > word2Length ? word1 : word2;
        return mergedEqualLength + longerWord.substring(minLength);
    }
//    public String mergeAlternately(String word1, String word2) {
//            int word1Length = word1.length();
//            int word2Length = word2.length();
//
//            StringBuilder sb = new StringBuilder();
//            for (int i = 0; i < word1Length; i++) {
//                sb.append(word1.charAt(i)); //could also do prev word1.toCharArray(); and get word1array[i]
//                if(i < word2Length) {
//                    sb.append(word2.charAt(i));
//                }
//            }
//            String mergedForFirstWord = sb.toString();
//            if (word2Length > word1Length) {
//                return mergedForFirstWord + word2.substring(word1Length);
//            }
//            return mergedForFirstWord;
//    }
}

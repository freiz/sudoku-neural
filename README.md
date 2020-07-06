There is a famous article [Solving Every Sudoku Puzzle](https://norvig.com/sudoku.html) which provided
a clean backtracking implementation to solve sudoku. It works great on almost all of the problems
however there are some cases the performance may be very bad because of trapped in too many searches
and recoveries.

```python
>>> hard1  = '.....6....59.....82....8....45........3........6..3.54...325..6..................'
>>> solve_all([hard1])
. . . |. . 6 |. . . 
. 5 9 |. . . |. . 8 
2 . . |. . 8 |. . . 
------+------+------
. 4 5 |. . . |. . . 
. . 3 |. . . |. . . 
. . 6 |. . 3 |. 5 4 
------+------+------
. . . |3 2 5 |. . 6 
. . . |. . . |. . . 
. . . |. . . |. . . 

4 3 8 |7 9 6 |2 1 5 
6 5 9 |1 3 2 |4 7 8 
2 7 1 |4 5 8 |6 9 3 
------+------+------
8 4 5 |2 1 9 |3 6 7 
7 1 3 |5 6 4 |8 2 9 
9 2 6 |8 7 3 |1 5 4 
------+------+------
1 9 4 |3 2 5 |7 8 6 
3 6 2 |9 8 7 |5 4 1 
5 8 7 |6 4 1 |9 3 2 

(188.79 seconds)

```
There are over 680k searches.

Another interesting article [Can Convolutional Neural Networks Crack Sudoku Puzzles?](https://github.com/Kyubyong/sudoku)
tried to predict the result using CNN and the result looks good, 86% on accuracy. While it is not a 
complete solution for sudoku problem since there is no guarantee a correct answer is returned.

Combine the two ideas together, we can use the prediction from CNN as a policy selector, i.e. at 
each step we will try the candidates one by one order by the confidence score from largest to smallest.
In the initial experiments, we reduced the searching steps from **680k** to **28** for the extreme case
and the running time reduced to 0.02s.

It is not a new idea, as early as 2015, the initial version of Alpha Go learns
human actions to build the policy network which used to aided the tree searching.

it is an application shows in real OR problem we do can leverage neural network
to guide how to search the space and have performance boost (though sudoku is too
easy that normally the search space is small so the performance boost is not significant
in most of the problems). Can neural network be used to solve mixed integer programming
in general to replace other human crafted heuristics practically?

### Reference
* https://norvig.com/sudoku.html
* https://github.com/Kyubyong/sudoku
* https://github.com/shivaverma/Sudoku-Solver
* https://www.kaggle.com/bryanpark/sudoku
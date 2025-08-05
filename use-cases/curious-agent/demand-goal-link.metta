(=( populate-demand-goal-link $space)
    add-reduce $space (superpose(
        (: (demand integrity 1.0) (Goal Confirm-Insight-Accuracy 0.9 0.6))
        (: (demand integrity 0.9) (Goal Confirm-Goal-Accuracy 0.9 0.6))
        (: (demand integrity 0.9) (Goal Confirm-Value-Accuracy 0.9 0.6))
        (: (demand integrity 0.9) (Goal Refine-Topic-Understanding 0.9 0.6))
        (: (demand integrity 1.0) (Goal Confirm-Conversation-Goal-Accuracy 0.9 0.6))
        (: (demand competence 0.9) (Goal Learn-Specific-Topic-Details 0.9 0.6))
        (: (demand competence 0.9) (Goal Understand-Topic-Insights 0.9 0.6))
        (: (demand competence 0.9) (Goal Evaluate-Solution-Feasibility 0.9 0.6))
        (: (demand competence 0.9) (Goal Learn-Goal-Details 0.9 0.6))
        (: (demand competence 0.9) (Goal Understand-Conversation-Goal-Relevance 0.9 0.6))
        (: (demand energy 0.9) (Goal Explore-Event-Emotions 0.9 0.6))
        (: (demand energy 0.9) (Goal Re-engage-User 0.9 0.6))
        (: (demand energy 0.9) (Goal Deepen-Topic-Exploration 0.9 0.6))
        (: (demand energy 0.9) (Goal Explore-Challenge-Reasons 0.9 0.6))
        (: (demand energy 0.9) (Goal Explore-Aspiration-Reasons 0.9 0.6))
        (: (demand certainty 0.9) (Goal Confirm-User-Engagement 0.9 0.6))
        (: (demand certainty 0.9) (Goal Confirm-Topic-Relevance 0.9 0.6))
        (: (demand certainty 0.9) (Goal Confirm-Solution-Accuracy 0.9 0.6))
        (: (demand certainty 0.9) (Goal Confirm-Aspiration-Accuracy 0.9 0.6))
        (: (demand certainty 1.0) (Goal Conclude-Conversation 0.9 0.6))
    ))
)
(= (get-goal-value (Goal $goal_name $goal_value $mean_stv))
    $goal_value
)
(= (find-best-goal_ $related_goals $max_value $best_goal)
    (if (== $related_goals ()) $best_goal 
        (let* (
            ($current-goal (car-atom $related_goals))
            ($next_goals (cdr-atom $related_goals))
            ($current_goal_value (get-goal-value $current-goal))
            ; (() println! (current-goal-value = $current_goal_value))
            (() (println! (currentgoal= $current-goal , current-goal-value= $current_goal_value , next-goals= $next_goals)))
        )
            (if (> $current_goal_value $max_value)
                (find-best-goal_ $next_goals $current_goal_value $current-goal)
                (find-best-goal_ $next_goals $max_value $best_goal)
            )
        )
    )
)

(= (find-best-goal ))
(= (goal-selector $space $demand)
    (let* (
        ($relatedGoals (collapse (get-type-space $space $demand)))
        ($selectedGoal (find-best-goal $relatedGoals))
    )
    $selectedGoal
)
)


; !(get-goal-value (Goal Confirm-Insight-Accuracy 0.9 0.6))
!(find-best-goal_ 
    ((Goal Confirm-Insight-Accuracy 0.5 0.6) 
     (Goal confirm-goal-accuracy 0.01 0.6) 
     (Goal refine-topic-understanding 0.1 0.6)) 
    -1 
    (Goal dummy 0 0)
  )
  
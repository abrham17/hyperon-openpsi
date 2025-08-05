(= (populate-link-space $space)
    (add-reduct $space (superpose(
        (: (demand integrity 1.0) (Goal Confirm-Insight-Accuracy 0.75 0.42))
        (: (demand integrity 1.0) (Goal Confirm-Goal-Accuracy 0.12 0.88))
        (: (demand integrity 1.0) (Goal Confirm-Value-Accuracy 0.91 0.33))
        (: (demand integrity 1.0) (Goal Refine-Topic-Understanding 0.05 0.67))
        (: (demand integrity 1.0) (Goal Confirm-Conversation-Goal-Accuracy 0.82 0.19))
        (: (demand competence 0.9) (Goal Learn-Specific-Topic-Details 0.37 0.95))
        (: (demand competence 0.9) (Goal Understand-Topic-Insights 0.64 0.21))
        (: (demand competence 0.9) (Goal Evaluate-Solution-Feasibility 0.99 0.50))
        (: (demand competence 0.9) (Goal Learn-Goal-Details 0.10 0.78))
        (: (demand competence 0.9) (Goal Understand-Conversation-Goal-Relevance 0.55 0.03))
        (: (demand energy 0.9) (Goal Explore-Event-Emotions 0.28 0.81))
        (: (demand energy 0.9) (Goal Re-engage-User 0.70 0.15))
        (: (demand energy 0.9) (Goal Deepen-Topic-Exploration 0.49 0.62))
        (: (demand energy 0.9) (Goal Explore-Challenge-Reasons 0.85 0.08))
        (: (demand energy 0.9) (Goal Explore-Aspiration-Reasons 0.17 0.73))
        (: (demand certainty 0.9) (Goal Confirm-User-Engagement 0.93 0.40))
        (: (demand certainty 0.9) (Goal Confirm-Topic-Relevance 0.02 0.89))
        (: (demand certainty 0.9) (Goal Confirm-Solution-Accuracy 0.77 0.25))
        (: (demand certainty 0.9) (Goal Confirm-Aspiration-Accuracy 0.31 0.66))
        (: (demand certainty 1.0) (Goal Conclude-Conversation 0.58 0.07))
    )))
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
            ; (() (println! (currentgoal= $current-goal , current-goal-value= $current_goal_value , next-goals= $next_goals)))
        )
            (if (> $current_goal_value $max_value)
                (find-best-goal_ $next_goals $current_goal_value $current-goal)
                (find-best-goal_ $next_goals $max_value $best_goal)
            )
        )
    )
)

(= (find-best-goal $goals)
    (find-best-goal_ $goals 0 ())
)
(= (goal-selector $space $demand)
    (let* (
        ($relatedGoals (collapse (get-type-space $space $demand)))
        (() (println! (related-goals= $relatedGoals)))
        ($selectedGoal (find-best-goal $relatedGoals))
    )
    $selectedGoal
)
)


; !(get-goal-value (Goal Confirm-Insight-Accuracy 0.9 0.6))
; !(find-best-goal_ 
;     ((Goal Confirm-Insight-Accuracy 0.5 0.6) 
;      (Goal confirm-goal-accuracy 0.01 0.6) 
;      (Goal refine-topic-understanding 0.1 0.6)) 
;     -1 
;     ()
;   )

; !(find-best-goal 
;     ((Goal Confirm-Insight-Accuracy 0.5 0.6) 
;      (Goal confirm-goal-accuracy 0.01 0.6) 
;      (Goal refine-topic-understanding 0.1 0.6)) 
   
;   )


;; unit test for goal selector
; !(bind! &linkSpace (new-space))

; !(populate-link-space &linkSpace)
; !(goal-selector &linkSpace (demand integrity 1.0))
; !(populate-link-space &linkSpace)
; !(goal-selector &linkSpace (demand integrity 1.0))

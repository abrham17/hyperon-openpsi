; Register and import modules
!(register-module! ../../../../hyperon-openpsi)
!(import! &self hyperon-openpsi:use-cases:curious-agent:demand-goal-link)

; Bind a new link space
!(bind! &linkSpace (new-space))


!(populate-link-space &linkSpace)

!(assertEqual (get-goal-value (Goal Confirm-Insight-Accuracy 0.9 0.6)) 0.9 )
!(assertEqual (find-best-goal' 
  ((Goal Confirm-Insight-Accuracy 0.5 0.6) 
   (Goal confirm-goal-accuracy 0.01 0.6) 
   (Goal refine-topic-understanding 0.1 0.6)) 
  -1 
  ()
) (Goal Confirm-Insight-Accuracy 0.5 0.6))

!(assertEqual (find-best-goal 
  ((Goal Confirm-Insight-Accuracy 0.5 0.6) 
   (Goal confirm-goal-accuracy 0.01 0.6) 
   (Goal refine-topic-understanding 0.91 0.6)) 
 
) (Goal refine-topic-understanding 0.91 0.6) )



!(assertEqual (goal-selector &linkSpace (demand integrity 1.0)) (Goal Confirm-Value-Accuracy 0.91 0.33) )


!(register-module! ../../../../hyperon-openpsi)
!(register-module! ../../../utilities-module)

!(import! &self hyperon-openpsi:main:modulator:modulator)
!(import! &self hyperon-openpsi:psi-utilities:psi_utils)
!(import! &self hyperon-openpsi:main:demand:demand)

!(import! &self hyperon-openpsi:main:mind-agents:modulator-updater:modulator-updaters)
!(import! &self hyperon-openpsi:main:mind-agents:modulator-updater:tests:modulator-updater-test-helpers)
!(import! &self hyperon-openpsi:main:mind-agents:action-planner:action-planner-v1)
!(import! &self hyperon-openpsi:main:mind-agents:action-planner:action-planner-v2)
!(import! &self hyperon-openpsi:main:mind-agents:feeling-updater:feeling-updaters)

!(import! &self utilities-module:utils)

!(bind! &kb (new-space))
!(bind! &modulator-space (new-space))
!(bind! &demandspace (new-space))
!(bind! &ruleSpace (new-space))

(= (addRulesToSpaceV2 $space) (
    add-reduct $space (superpose (
        (addGoal (found_target) 0.5 () $space)
        (addGoal (init) 0 1 $space)
        (addGoal (hold_item) 1 1 $space)
        (addRule $space r1 ((init)) (explore) (found_target) (TTV 0 (STV 1.0 1.0)))
        (addRule $space r2 ((found_target)) (pick_item) (hold_item) (TTV 0 (STV 1.0 1.0)))
    ))
))

;; Populate the spaces
!(insert-modulators &modulator-space)
!(addRulesToSpaceV2 &ruleSpace)
!(addDemand &demandspace energy 0.6)
!(addDemand &demandspace affiliation 0.5)
!(addDemand &demandspace competence 0.2)

(= (extractGoalFromRuleV2 $rule)
    (let (: $handle (IMPLICATION_LINK (AND_LINK ($context $action)) $goal)) $rule $goal)
)

(= (findRulesWithActionV2 $space $action)
    (collapse (match $space (: $handle (IMPLICATION_LINK (AND_LINK ($context $action)) $goal))
        (: $handle (IMPLICATION_LINK (AND_LINK ($context $action)) $goal))
    ))
)

(=(reverse $list $acc)  
    (if(== $list ()) $acc
    (let ($head $tail ) (decons-atom $list) (reverse $tail (cons-atom $head $acc) ))
    )
)

(: remove-atoms' (-> hyperon::space::DynSpace Expression Expression))
(= (remove-atoms' $space $expression)
    (if (== $expression ())
        ()
        (let* (
            (($head $tail) (decons-atom $expression))
            ($_  (remove-atom $space $head))
           ($rem (collapse (get-atoms $space)))
           ($_ (println! (remaining-atoms $rem)))
        )
            (remove-atoms' $space $tail)
        )
    )
)

;; This function removes values from a given atomspace
;;(:remove-atoms (-> hyperon::space::DynSpace))
(= (remove-atoms $space)
    (let $atoms (collapse (get-atoms $space))
        (remove-atoms' $space $atoms)
    )
 )

;; Remove The remove occurrences function removes values based on a given 
;;pattern for instance a perception updater value can be stored as 
;;(perception value time)
(: remove-occurrences (-> hyperon::space::DynSpace Expression Expression)) 
(= (remove-occurrences $space $pattern)
   	(let* (
            ($atoms (collapse (match $space $pattern $pattern)))
            ($_ (println! (atoms-to-remove $atoms)))
           ) 
	  	(remove-atoms' $space $atoms)))


(= (add-atoms $space $values)
	(add-reduct $space (superpose $values)))


;;The below function is used to synchronize two atomspaces for having the same
;;Replicated value. The updater space belongs to the mind-agent while the atomspace to be synchronized is the 
;; knowledge base atomspace.
(: update-atoms (-> Expression hyperon::space::DynSpace hyperon::space::DynSpace Expression ))
(= (update-atoms $pattern $updater $updated)
   	(let* (
	       ($updater-atoms (collapse (get-atoms $updater)))
	       ($rem-res (remove-occurrences $updated $pattern))
            ($res (collapse (add-atoms $updated $updater-atoms)))
	       )
	  	()	
	)
)



;;Tag based specification for atomspace updation
(: update-all (-> Symbol hyperon::space::DynSpace hyperon::space::DynSpace Expression))
(= (update-all $tag $updater $updated)
   (case $tag(
              (demand (update-atoms (demand $x $y) $updater $updated))
              (modulator (update-atoms (modulator $x $y) $updater $updated))
              )
))



;; a querying mechanism of the knowledge base.
(= (query $space ((: $handle ((TTV $time (STV $bel $conf)) (IMPLICATION_LINK (AND_LINK ((Goal $context $x $y) $action)) $goal))) $cumm-weight))
   	(collapse (match $space ((: $handle ((TTV $time (STV $bel $conf)) (IMPLICATION_LINK (AND_LINK ((Goal $context $x $y) $action)) $goal))) $cumm-weight)
	       ((: $handle ((TTV $time (STV $bel $conf)) (IMPLICATION_LINK (AND_LINK ((Goal $context $x $y) $action)) $goal))) $cumm-weight))

))
(= (query $space (modulator $x $y)) (collapse (match $space (modulator $x $y) (modulator $x $y))))
(= (query $space (perception $x $y)) (collapse (match $space (perception $x $y) (perception $x $y))))
(= (query $space (demand $x $y)) (collapse (match $space (demand $x $y) (demand $x $y))))

(: queryAll (-> hyperon::space::DynSpace Symbol Expression))
(= (queryAll $space $type)
   (case $type(
	       (modulator (query $space (modulator $x $y)))
	       (perception (query $space (perception $x $y)))
	       (demand (query $space (demand $x $y)))
	       (rule (query $space ((: $handle ((TTV $time (STV $bel $conf)) (IMPLICATION_LINK (AND_LINK ((Goal $context $x $y) $action)) $goal))) $cumm-weight)))
	       ($_ (Error "This form doesn't exist in the atomspace"))
)))


;; The demands are represented based on their satifsaction, 
;; a lower satisfation will require urgent response.
(: demandSelector' (-> Expression Number Demand))
(= (demandSelector' $demands $maxDemand)
   	(if (== $demands ()) 
	  	$maxDemand
		(let* (
		       	((demand $maxName $maxVal) $maxDemand)
		       	(((demand $currName $currVal) $tail) (decons-atom $demands))
		       )
		  	(if (<= $currVal $maxVal)
			  	(demandSelector' $tail (demand $currName $currVal))
				(demandSelector' $tail $maxDemand)
			)
		)
	)
)

;; (: demandSelector (-> Expression Demand))
(= (demandSelector $demands)
   	(if (== $demands ())
	  	(Error "No Demands Available")
		(demandSelector' $demands (car-atom $demands))))


(= (goalSelector $demand)
   (case $demand(
		 ((demand affiliation $y) (Goal mission_complete 1.0 1.0))
		((demand competence $x) (Goal mission_complete 1.0 1.0))
        ((demand energy $x) (Goal mission_complete 1.0 1.0))
	)
))


(= (applyActions $actions $goal $demand) 
   (if (== $actions ())
     		()
		(let $isGoalMet? (isGoalMet? $actions $goal)
		  	(if $isGoalMet?
                (let* (
                       ($_ (println! (goalMet $isGoalMet?)))
                       ((demand $demandName $demandValue) $demand)
					   ($newDemValue (+ $demandValue (* (- 1 $demandValue) 0.5)))
                )
			  	    (setDemandValue &demandspace $demandName $newDemValue)
                )
				()
			)
)))

; A non-deterministic implementation of demand selection refer 
;; OpenPsi: A Novel Computational Affective Model and its Application in Video Games by Goertzel,Ben et. al (2013)
(: selectDemand (-> Expression Expression))
(= (selectDemand $demands)
   (let* (
	  ($selectionVal (fetch-modulator-val &modulator-space selection-threshold))
	  ($_ (println! (This is the selection thresholdvalue $selectionVal)))
	  ($randomVal (random-float $rng 0 1))
	)
     	(if (>= $selectionVal $randomVal)
	  	(demandSelector $demands)
		(let* (
			($length (size-atom $demands))
			($randIndex (random-int &rng 0 (- $length 1)))
			($demand (index-atom $demands $randIndex))
		)
		  $demand)
)))

(: selectRuleByAction (-> hyperon::space::DynSpace Symbol Expression)) 
(= (selectRuleByAction $ruleSpace $action)
   	(collapse (match $ruleSpace ((:$handle ((TTV $time (STV $bel $conf)) (IMPLICATION_LINK (AND_LINK ($context $action)) $goal))) $val) ((: $handle ((TTV $time (STV $bel $conf)) (IMPLICATION_LINK (AND_LINK ($context $action)) $goal))) $val))))

;; A temporary selector of rules which are selected based on having the same actions. 
(: selectMaxRule (-> Expression Expression Expression))
(= (selectMaxRule $rules $maxRule)
   	(if (== $rules ())
	  $maxRule
	(let* (
	       ((($rule $weight) $tail) (decons-atom $rules))
		    (($maxRule' $maxWeight) $maxRule)
	       )
	  (if (>= $maxWeight $weight)
			(selectMaxRule $tail $maxRule)
			(selectMaxRule $tail $rule)
		)
)))





(:isGoalMet' (-> Expression Goal Bool))
(= (isGoalMet' $rules $goal)
   	(if (== $rules ())
		False
		(let* (
			(($rule $tail) (decons-atom $rules))
            ;($_ (println! (isGoalMetInnerRule $rule tail $tail)))
			($ruleGoal (extractGoalFromRuleV2 $rule))
			((Goal $g $tv $sgv) $goal)
            ($_ (println! (this is the goal $ruleGoal = $g)))
		)
		  (if (== $ruleGoal $goal)
		    	True
		    	(isGoalMet' $tail $goal)
		  )
		)
	)
)

(: isGoalMet? (-> Expression Goal Bool))
(= (isGoalMet? $actions $goal)
	(let* (
	       ($length (size-atom $actions))
	       ($lastAction (index-atom $actions (- $length 1)))
	       ($rules' (findRulesWithAction &ruleSpace $lastAction))
           ($_ (println! (rulesinGoal $rules')))
	)
		 (isGoalMet' $rules' $goal)
	)	
)

;; This function updates the modulator values in the atomspace.
(:update-modulator (-> Symbol Number Expression))
(= (update-modulator $modulatorName $modulatorValue)
	(let* (
	       ($modulator (fetch-modulator &modulator-space $modulatorName))
	       ($_ (update-atom &modulator-space $modulator ($modulatorName $modulatorValue)))
	)
	  	()
	)
)


;; This function implements the kb-initializer for the open-psi loop to work
(: kb-initializer (-> hyperon::space::DynSpace Expression))
(= (kb-initializer $kb)
   (let* (
	  	 ($rules (collapse (get-atoms &ruleSpace)))
		($modulators (collapse (get-atoms &modulator-space)))
		($demands (collapse (get-atoms &demandspace)))
		($partialKb (union-atom $rules $modulators))
		($kbExp (union-atom $partialKb $demands))
		($_ (collapse (add-atoms $kb $kbExp)))
	)
     		()
    
	)
)

(: psi-loop-v2 (-> Symbol hyperon::space::DynSpace Expression)) 
(= (psi-loop-v2 $state $kb)
	(let* (
	    ($demands (queryAll $kb demand))
        ($_ (println! (Available Demands $demands)))

		($currDemand (demandSelector $demands))
        ($_ (println! (Selected Demand: $currDemand)))

        ($goalObj (goalSelector $currDemand))
		((Goal $goal $goalVal1 $goalVal2) $goalObj) 
        ($_ (println! (Selected Goal: $goal)))

		($actions (hillClimbingPlanner $state $goal (Tested Actions) () &ruleSpace))
        ($_ (println! (Actions: $actions goal $goal demand $currDemand)))

		($applyValue (applyActions $actions $goalObj $currDemand)) 
        ($_ (println! (applyValue $applyValue)))

		($competenceDemand (getDemandByName &demandspace competence)) 
		($affiliationDemand (getDemandByName &demandspace affiliation))
		($energyDemand (getDemandByName &demandspace energy))
        ($_ (println! (Demand Values: $competenceDemand $affiliationDemand $energyDemand)))
		
		($modUpdate (modulatorUpdaterAgent &modulator-space $competenceDemand $affiliationDemand $energyDemand))
        ($_ (println! (modUpdate $modUpdate modulatorValues (collapse (get-atoms &modulator-space))))) 
		
		($happinessValue (happinessFeelingUpdater &modulator-space))
		($sadnessValue (sadnessFeelingUpdater &modulator-space))
		($angerValue (angerFeelingUpdater &modulator-space))
		($fearValue (fearFeelingUpdater &modulator-space))
		($excitementValue (excitementFeelingUpdater &modulator-space))
		($loveValue (loveFeelingUpdater &modulator-space))
		($hateValue (hateFeelingUpdater &modulator-space))
		($gratitude (gratitudeFeelingUpdater &modulator-space))
	    
		($_ (update-all modulator &modulator-space $kb))
        ($_ (update-all demand &demandspace $kb))
	)
	  (happinessValue $happinessValue sadnessValue $sadnessValue angerValue $angerValue fearValue $fearValue gratitudeValue $gratitude) 
	)	
)

(= (isListEmpty $list)
	(if (== (size-atom $list) 0) True False)
)

(= (emotionChecker $emotion)
	(unify $emotion (happinessValue $happinessValue sadnessValue $sadnessValue angerValue $angerValue fearValue $fearValue gratitudeValue $gratitude) True False)
)


!(assertEqual (kb-initializer &kb) ())
!(assertEqual (isListEmpty (queryAll &kb modulator)) False)
!(assertEqual (isListEmpty (queryAll &kb rule)) False)
!(assertEqual (isListEmpty (queryAll &kb demand)) False)
!(assertEqual (isGoalMet? (report) (Goal mission_complete 1.0 1.0)) True)
!(assertEqual (isGoalMet? (analyze upload) (Goal analyze_data 1 1)) False)
!(assertEqual (emotionChecker (psi-loop-v2 init &kb)) True)
!(assertEqual (demandSelector ((demand Affiliation 3) (demand Concept 4) (demand Car 8))) (demand Affiliation 3))

;;!(kb-initializer &kb)
;; !(psi-agent-loop init &kb)


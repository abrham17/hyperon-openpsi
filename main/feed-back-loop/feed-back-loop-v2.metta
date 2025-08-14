(: psi-loop (-> Symbol hyperon::space::DynSpace Expression)) 
(= (psi-loop $state $kb)
	(let* (
	    ($demands (queryAll $kb demand))
        ($_ (println! (demands $demands)))

		($currDemand (demandSelector $demands))
        ($_ (println! (dmSelector $currDemand)))

        ($goalObj (goalSelector $currDemand))
		((Goal $goal $goalVal1 $goalVal2) $goalObj) 
        ($_ (println! (goal-psi $goal)))

		($actions-reversed (hillClimbingPlanner $state $goal (TestedActions) () &ruleSpace)) ;; Hillclimber returns actions in reverse order.
		($actions (reverse $actions-reversed ()))
        ($_ (println! (actions-psi-to-apply $actions goal $goal demand $currDemand)))

		($applyValue (applyActions $actions $goalObj $currDemand)) 
        ($_ (println! (applyValue $applyValue)))

		($competenceDemand (getDemandByName &demandspace competence)) 
		($affiliationDemand (getDemandByName &demandspace affiliation))
		($energyDemand (getDemandByName &demandspace energy))
        ($_ (println! (demandValues $competenceDemand $affiliationDemand $energyDemand)))
		
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
	  (hateValue $hateValue happinessValue $happinessValue sadnessValue $sadnessValue angerValue $angerValue)
	)	
)
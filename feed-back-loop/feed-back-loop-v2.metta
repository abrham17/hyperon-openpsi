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

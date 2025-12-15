import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
import pandas as pd
from forecasting.global_forecasting import train_and_forecast_global


logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for global LightGBM forecasting.
    Replaces the original per-pool XGBoost approach with global model.
    """
    logger.info("Starting global LightGBM forecasting for all pools (replacing per-pool XGBoost)")
    
    try:
        result = train_and_forecast_global(
            pool_ids=None,  # Use all filtered pools
            train_days=60,
            forecast_ahead=1,
            use_tvl_stacking=True,
            n_trials=5
        )
        
        # Print comprehensive summary
        logger.info("\n" + "="*60)
        logger.info("📊 GLOBAL LIGHTGBM FORECASTING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total pools processed: {result['total_pools']}")
        logger.info(f"✅ Model-based forecasts: {result['model_pools']}")
        logger.info(f"🟡 Cold-start baselines: {result['cold_start_pools']}")
        logger.info(f"📈 Success rate: {(result['model_pools']/result['total_pools']*100):.1f}%")
        logger.info(f"🔧 TVL stacking: {'Enabled' if result['use_tvl_stacking'] else 'Disabled'}")
        logger.info(f"🧠 APY features: {result['apy_feature_count']}")
        logger.info(f"💰 TVL features: {result['tvl_feature_count']}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Fatal error in global forecasting: {e}")
        raise

if __name__ == "__main__":
    main()